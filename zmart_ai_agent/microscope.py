"""The microscope, as the agent holds it: a ZMART Controller session and what it learned.

The agent knows nothing about a microscope in advance. When it connects,
it asks the controller's own commands what this microscope is and what it can
do: ``get_info`` (the driver's description in plain words, and where images
go), ``get_actuators`` and ``get_xyz`` (the axes, their motors and how far
they travel), ``get_state`` (the settings that can be changed, and the
read-only report), ``get_acquisition_settings`` and ``get_procedures``. From
the answers it writes the "This microscope" section of the model's
instructions (``instrument_section``). The controller is not shaped around
the agent: these are the calls every ZMART driver answers anyway.

``Microscope`` also holds the window's side of the conversation: where images
and warnings go, the vision model, the schedules, the plans and the
go-ahead bookkeeping that ``tools.py`` uses.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import contextlib
import inspect
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from zmart_controller import utils
from zmart_controller.session import Session, set_instrument

from .eyes import Eyes
from .instructions import INSTRUMENT_SECTION, NO_DESCRIPTION, NOT_CONNECTED, UNANSWERED
from .schedules import Scheduler
from .settings import CLOCK_FORMAT, MODEL

# The commands the agent learns a microscope from, in the order it asks them.
LEARNED_FROM = {
    "info": "get_info",
    "actuators": "get_actuators",
    "xyz": "get_xyz",
    "state": "get_state",
    "acquisition_settings": "get_acquisition_settings",
    "procedures": "get_procedures",
}
NOT_CHOSEN = "no microscope is chosen"


def identity(instrument: dict[str, Any]) -> dict[str, str]:
    """The three names of an instrument (vendor, microscope, api), and nothing else.

    A connection dictionary may hold a password or a login, so only these
    three keys are ever shown to the model or the operator.
    """
    return {key: instrument[key] for key in utils.IDENTITY}


@dataclass
class Microscope:
    """One microscope through the ZMART Controller, plus the window's side of the conversation.

    ``instrument`` is one of the dictionaries ``zmart_controller.get_instruments()``
    lists (or None when none is chosen yet); ``connect`` opens a session on it
    and learns the microscope.
    """

    instrument: dict[str, Any] | None = None
    on_image: Callable[[np.ndarray, str], None] = lambda image, caption: None
    on_warning: Callable[[str], None] = lambda text: None
    on_tool: Callable[[str, dict], None] = lambda name, args: None  # each tool call, as it starts
    vision_model: Any = MODEL  # a model name, a model object, or a test model
    vision: bool = True  # False when the vision model cannot be shown images
    _eyes: Eyes | None = field(default=None, repr=False)  # see the ``eyes`` property
    scheduler: Scheduler = field(default_factory=Scheduler)  # what is to happen later
    # Whether a reply that called no tool is challenged once (see the guards in tools.py).
    # Off for scripted tests, on in the window.
    challenge_no_tool: bool = False
    plans: dict[str, Any] = field(default_factory=dict)  # plan id -> AcquisitionPlan
    planned_in: dict[str, int] = field(default_factory=dict)  # plan id -> turn it was last shown
    # Set by Cancel or Stop: every further tool call in this turn does nothing, and a
    # running acquisition ends after the image being taken.
    cancel: threading.Event = field(default_factory=threading.Event)
    # Where the stage was when the operator last wrote, or where they last agreed
    # to go. Moves are measured from here, so small steps cannot add up to a long
    # move unasked.
    anchor: dict[str, float] | None = None
    turn: int = 0  # the operator's messages so far
    # Long moves, routines and acquisitions the agent asked the operator about, and when.
    go_ahead_asked: dict[str, int] = field(default_factory=dict)
    session: Session | None = field(default=None, repr=False)
    learned: dict[str, Any] | None = None  # the answers read at connect, by LEARNED_FROM key
    connect_error: str | None = None
    # The agent's thread and the window's both read the microscope; one call at a time.
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # -- the connection -------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The microscope's three names, as "vendor / microscope / api"."""
        return " / ".join(identity(self.instrument).values()) if self.instrument else NOT_CHOSEN

    def connect(self) -> None:
        """Open a session on the chosen instrument and learn the microscope from it.

        An open session is closed first, so this also reconnects. A failure is
        kept in ``connect_error`` (for check_setup and the window) and raised.
        """
        self.disconnect()
        if self.instrument is None:
            self.connect_error = NOT_CHOSEN
            raise RuntimeError(f"{NOT_CHOSEN}; choose one in the window or with --instrument")
        try:
            session = set_instrument(self.instrument)
        except Exception as exc:
            self.connect_error = f"{type(exc).__name__}: {exc}"
            raise
        with self._lock:
            self.session, self.connect_error = session, None
            self.learned = learn(session)

    def ensure_connected(self) -> None:
        """Connect when no session is open (the first message, or after a failed connect)."""
        if self.session is None:
            self.connect()

    def disconnect(self) -> None:
        """Close the session, if one is open. A driver that fails to close is let go anyway."""
        with self._lock:
            session, self.session, self.learned = self.session, None, None
        if session is not None:
            with contextlib.suppress(Exception):
                session.disconnect()

    def call(self, command: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """One controller command; the whole answer, ``{"success", "content"}``.

        The driver's refusals pass through unchanged: ValueError for a request
        that is wrong, RuntimeError for a failure at the microscope.
        """
        with self._lock:
            if self.session is None:
                reason = self.connect_error or NOT_CHOSEN
                raise RuntimeError(f"no microscope is connected ({reason})")
            return getattr(self.session, command)(*args, **kwargs)

    def read(self, command: str, **kwargs: Any) -> Any:
        """The content of a reading command; a reading the driver could not make raises."""
        answer = self.call(command, **kwargs)
        if not answer.get("success"):
            content = json.dumps(answer.get("content"), default=str)
            raise RuntimeError(f"{command} did not succeed: {content}")
        return answer["content"]

    # -- readings --------------------------------------------------------------------------

    def position(self) -> dict[str, float]:
        """Where the stage is: x, y and z in the driver's frame."""
        xyz = self.read("get_xyz")
        return {axis: xyz[axis]["value"] for axis in ("x", "y", "z")}

    def where(self) -> dict[str, Any]:
        """The position and the settings: what a picture depends on."""
        state = self.read("get_state")
        return {"position_um": self.position(), "settings": state.get("changeable")}

    def state(self) -> dict[str, Any]:
        """A compact picture of the microscope, sent with every message from the operator."""
        state = self.read("get_state")
        return {
            "position_um": self.position(),
            "settings": state.get("changeable"),
            "observed": state.get("observed"),
            "clock": time.strftime(CLOCK_FORMAT),
            "schedules": self.scheduler.listing(),
        }

    @property
    def has_description(self) -> bool:
        """Whether the driver describes the microscope in words (get_info's description)."""
        info = (self.learned or {}).get("info")
        description = info.get("description") if isinstance(info, dict) else None
        return isinstance(description, str) and bool(description.strip())

    def instrument_section(self) -> str:
        """The "This microscope" part of the model's instructions, from what was learned."""
        if self.learned is None:
            return NOT_CONNECTED.format(reason=self.connect_error or NOT_CHOSEN)
        return instrument_section(self.name, self.learned)

    def driver_folder(self) -> Path | None:
        """The folder of the chosen microscope's driver, as the controller registered it.

        The controller keeps the driver's functions; the file they were loaded
        from shows where the driver lives. A driver laid out as the controller
        describes keeps them in a ``zmart_controller`` folder inside it.
        """
        if self.instrument is None:
            return None
        entry = utils.REGISTRY.get(tuple(identity(self.instrument).values()))
        if entry is None:
            return None
        try:
            plugin = Path(inspect.getfile(entry["ops"]["get_info"])).resolve().parent
        except (TypeError, OSError):  # a function without a file, such as a built-in
            return None
        return plugin.parent if plugin.name == utils.PLUGIN_FOLDER else plugin

    # -- the conversation's side -------------------------------------------------------------

    @property
    def eyes(self) -> Eyes:
        """The vision model's own conversation, built from ``vision_model`` on first use.

        When ``vision_model`` is changed, the next look gets new eyes, so the
        model in use is always the one the eyes talk to; the images seen with
        the old one are forgotten, since another model cannot read its turns.
        """
        if self._eyes is None or self._eyes.model is not self.vision_model:
            self._eyes = Eyes(self.vision_model)
        return self._eyes

    @eyes.setter
    def eyes(self, eyes: Eyes) -> None:
        self._eyes, self.vision_model = eyes, eyes.model

    def stop(self) -> None:
        """Cancel the agent's turn, end a running acquisition, and drop every schedule.

        A running acquisition ends after the image being taken. A single move
        or image the driver has already started runs to its end: the ZMART
        vocabulary has no command to interrupt it, so the microscope's own
        controls stop it sooner. The schedules go too, or one could start the
        microscope again a moment after Stop was pressed.
        """
        self.cancel.set()
        self.scheduler.clear()


def learn(session: Session) -> dict[str, Any]:
    """Ask the driver every reading command once; the answers by LEARNED_FROM key.

    A command that fails, or answers success false, is kept as its reason
    under ``unanswered``, so one missing answer never stops the connection:
    the section then says what the driver did not answer.
    """
    learned: dict[str, Any] = {"unanswered": {}}
    for key, command in LEARNED_FROM.items():
        try:
            answer = getattr(session, command)()
        except Exception as exc:
            learned["unanswered"][key] = f"{type(exc).__name__}: {exc}"
            continue
        if isinstance(answer, dict) and answer.get("success"):
            learned[key] = answer.get("content")
        else:
            learned["unanswered"][key] = f"{command} answered {json.dumps(answer, default=str)}"
    return learned


def instrument_section(name: str, learned: dict[str, Any]) -> str:
    """The section of the instructions about one microscope, in the driver's own words."""
    unanswered = learned["unanswered"]

    def missing(*keys: str) -> str | None:
        reasons = [unanswered[key] for key in keys if key in unanswered]
        return UNANSWERED.format(reason="; ".join(reasons)) if reasons else None

    info = learned.get("info") or {}
    description = info.get("description")
    if not (isinstance(description, str) and description.strip()):
        description = missing("info") or NO_DESCRIPTION
    state = learned.get("state") or {}
    return INSTRUMENT_SECTION.format(
        name=name,
        description=description.strip(),
        output_root=info.get("output_root") or missing("info") or "(not given)",
        axes=missing("xyz", "actuators") or _axes(learned["xyz"], learned["actuators"]),
        settings=missing("state") or json.dumps(state.get("changeable"), default=str),
        observed=missing("state") or json.dumps(state.get("observed"), default=str),
        acquisition_settings=missing("acquisition_settings")
        or _listed(learned["acquisition_settings"], _acquisition_setting),
        procedures=missing("procedures") or _listed(learned["procedures"], _procedure),
    )


def _axes(xyz: dict[str, Any], actuators: dict[str, Any]) -> str:
    lines = []
    for axis, reading in xyz.items():
        low, high = reading.get("range") or (None, None)
        unit = reading.get("unit", "")
        span = f"from {low:g} to {high:g} {unit}" if None not in (low, high) else "range not given"
        motors = ", ".join(str(m) for m in actuators.get(axis, [reading.get("actuator")]))
        lines.append(f"  {axis}: {span}; motors: {motors}")
    return "\n".join(lines)


def _listed(entries: dict[str, Any], line: Callable[[str, Any], str]) -> str:
    return "\n".join(f"  {line(name, spec)}" for name, spec in entries.items()) or "  (none)"


def _acquisition_setting(name: str, spec: Any) -> str:
    if not isinstance(spec, dict):
        return f"{name}: {json.dumps(spec, default=str)}"
    allowed = spec.get("options")
    allowed = json.dumps(allowed) if isinstance(allowed, list) else str(allowed)
    return f"{name}: allowed {allowed}; active {json.dumps(spec.get('active'), default=str)}"


def _procedure(name: str, spec: Any) -> str:
    description = spec.get("description") if isinstance(spec, dict) else spec
    return f"{name}: {description}"
