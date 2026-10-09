"""The microscope, as the agent holds it: a ZMART Controller session and what it learned.

The agent knows nothing about a microscope in advance. When it connects,
it asks the controller's own commands what this microscope is and what it can
do: ``get_info`` (the driver's description in plain words, and where images
go), ``get_actuators`` and ``get_xyz`` (the axes, their motors, and the
canvas: everywhere a picture can show), ``get_state`` (the settings that can be changed, and the
read-only report), ``get_acquisition_settings`` and ``get_procedures``. From
the answers it writes the "This microscope" section of the model's
instructions (``instrument_section``). The controller is not shaped around
the agent: these are the calls every ZMART driver answers anyway.

A driver is named the way the controller lists it: ``"mock"`` is the
simulated microscope that comes with the controller, and any driver installed
on this computer with ``zmart_controller.register_driver`` by the name in its
``zmart_driver.json``. A driver module or class can also be handed over
directly, which is what the tests do.

``Microscope`` also holds the window's side of the conversation: where images
and warnings go, the vision model and its eyes, the frames seen, the
schedules, the requests, the plans and the go-ahead bookkeeping that the tools
use. Its one clock is the scheduler's: every time the agent writes comes from
it, so a test can decide when time passes.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import contextlib
import inspect
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from zmart_controller import ZmartController
from zmart_controller.registry import driver_name, find_driver, get_instruments

from .eyes import Eyes
from .frames import Calibration, FrameHistory, sample_map
from .instructions import INSTRUMENT_SECTION, NO_DESCRIPTION, NOT_CONNECTED, UNANSWERED
from .requests import Requests
from .schedules import Scheduler, hms
from .settings import DEFAULT_DRIVER, MODEL

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
AXES = ("x", "y", "z")


def instruments() -> list[str]:
    """The drivers this computer offers, by name: the simulated microscope first, then
    every driver installed with the controller."""
    return [DEFAULT_DRIVER, *(name for name in get_instruments() if name != DEFAULT_DRIVER)]


@dataclass
class Microscope:
    """One microscope through the ZMART Controller, plus the window's side of the conversation.

    ``driver`` is the microscope's ZMART driver: its name in the controller's
    list (``"mock"``, or an installed driver's name), or a driver module or
    class, or None when none is chosen yet. ``connection`` is handed to the
    driver as it is, for whatever that driver needs, such as a host name; it
    adds to (and overrides) the connection saved with an installed driver.
    It may hold a password, so it is never shown to the model or the
    operator. ``connect`` opens a session and learns the microscope.
    """

    driver: Any = None
    connection: dict[str, Any] | None = None
    on_image: Callable[[np.ndarray, str], None] = lambda image, caption: None
    on_warning: Callable[[str], None] = lambda text: None
    on_tool: Callable[[str, dict], None] = lambda name, args: None  # each tool call, as it starts
    vision_model: Any = MODEL  # a model name, a model object, or a test model
    vision: bool = True  # False when the vision model cannot be shown images
    _eyes: Eyes | None = field(default=None, repr=False)  # see the ``eyes`` property
    scheduler: Scheduler = field(default_factory=Scheduler)  # what is to happen later
    # Whether a reply that called no tool is challenged once (see guards.py).
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
    # What the model last saw of the position and settings, for state_changed in
    # every acting tool's answer (see tooling.StateTrail).
    seen: dict[str, Any] | None = None
    session: ZmartController | None = field(default=None, repr=False)
    learned: dict[str, Any] | None = None  # the answers read at connect, by LEARNED_FROM key
    connect_error: str | None = None
    # The agent's thread and the window's both read the microscope; one call at a time.
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    frames: FrameHistory = field(init=False, repr=False)  # the images seen this session
    requests: Requests = field(init=False, repr=False)  # what each typed message set going

    def __post_init__(self) -> None:
        # Both tell the time through ``now``, so a clock handed to the scheduler
        # later (as the tests do) reaches them too.
        self.frames = FrameHistory(self.now, calibration=Calibration(), name=self.name)
        self.requests = Requests(self.now)

    # -- the clock ----------------------------------------------------------------------

    def now(self) -> float:
        """The agent's one clock: the scheduler's, in seconds since the epoch."""
        return self.scheduler.clock()

    # -- the connection -------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The driver's name, such as "mock"."""
        if self.driver is None:
            return NOT_CHOSEN
        return self.driver if isinstance(self.driver, str) else driver_name(self.driver)

    def connect(self) -> None:
        """Plug in the chosen driver, connect, and learn the microscope from it.

        An open session is closed first, so this also reconnects. A failure is
        kept in ``connect_error`` (for check_setup and the window) and raised.
        """
        self.disconnect()
        if self.driver is None:
            self.connect_error = NOT_CHOSEN
            raise RuntimeError(f"{NOT_CHOSEN}; choose a driver in the window or with --driver")
        try:
            session = ZmartController(self.driver, self._connection())
        except Exception as exc:
            self.connect_error = f"{type(exc).__name__}: {exc}"
            raise
        with self._lock:
            self.session, self.connect_error = session, None
            self.learned = learn(session)
            self.frames.name = self.name  # the calibration is kept per microscope

    def _connection(self) -> dict[str, Any] | None:
        """What the driver is connected with: for an installed driver, the connection
        saved with it, with what was given here added on top."""
        if isinstance(self.driver, str) and self.connection:
            return {**find_driver(self.driver)[1], **self.connection}
        return self.connection

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

        The controller never raises: when a driver's method raises, it answers
        success False with the error's name in front of its message
        ("ValueError: unknown actuator 'hydraulic'"). The agent unfolds such
        an answer into the error again, so that a refusal (a ValueError: the
        request was wrong) and a failure at the microscope (any other error)
        are told apart, as the tools expect. A soft answer, success False
        with the driver's own words or a dictionary, is handed back as it is.
        """
        with self._lock:
            if self.session is None:
                reason = self.connect_error or NOT_CHOSEN
                raise RuntimeError(f"no microscope is connected ({reason})")
            answer = getattr(self.session, command)(*args, **kwargs)
        if (error := folded_error(answer)) is not None:
            raise error
        return answer

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
        return {axis: xyz[axis]["position"] for axis in AXES}

    def snapshot(self) -> dict[str, Any]:
        """The position, the settings and the read-only report, in one reading."""
        state = self.read("get_state")
        return {
            "position_um": self.position(),
            "settings": state.get("changeable"),
            "observed": state.get("observed"),
        }

    def where(self) -> dict[str, Any]:
        """The position and the settings: what a picture depends on."""
        snapshot = self.snapshot()
        return {"position_um": snapshot["position_um"], "settings": snapshot["settings"]}

    def state(self, origin: str = "operator") -> dict[str, Any]:
        """A compact picture of the microscope and the session, sent with every message.

        The position, the settings and the observed report; the clock and the
        schedules; the frames seen so far and the map made from them, once
        there are any; and the request this turn belongs to, when the turn was
        written by the machine (``origin`` "scheduled" or "continuation") or
        the request has a plan or a wait.
        """
        state = self.snapshot()
        state["clock"] = hms(self.now())
        state["schedules"] = self.scheduler.listing()
        if (listing := self.frames.listing()) is not None:
            state["frames"] = listing
            state["map"] = sample_map(self.frames)
        request = self.requests.current
        if request is not None and (origin != "operator" or request.plan or request.wait):
            state["request"] = request.brief(self.now())
        return state

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
        """The folder of the chosen driver, for the source-reading tools.

        An installed driver's folder is where its ``zmart_driver.json`` and class
        file live; a driver module that is a package gives its folder. A driver
        that is a single file outside any package gives none: the folder it
        sits in may hold much else, and the agent should read the driver, not
        that.
        """
        driver = self.driver
        try:
            if isinstance(driver, str):
                driver = find_driver(driver)[0]
            if hasattr(driver, "__path__"):  # a package
                return Path(inspect.getfile(driver)).resolve().parent
            driver_class = (
                driver if isinstance(driver, type) else getattr(driver, "ZmartDriver", None)
            )
            if driver_class is not None:
                return Path(inspect.getfile(driver_class)).resolve().parent
        except (TypeError, OSError, ValueError):  # no file to be found
            return None
        return None

    # -- the conversation's side -------------------------------------------------------------

    @property
    def eyes(self) -> Eyes:
        """The vision model's own conversation, built from ``vision_model`` on first use.

        When ``vision_model`` is changed, the next look gets new eyes, so the
        model in use is always the one the eyes talk to; the looks of the old
        one are forgotten, since another model cannot read its turns.
        """
        if self._eyes is None or self._eyes.model is not self.vision_model:
            self._eyes = Eyes(self.vision_model, clock=self.now)
        return self._eyes

    @eyes.setter
    def eyes(self, eyes: Eyes) -> None:
        self._eyes, self.vision_model = eyes, eyes.model

    def stop(self) -> None:
        """Cancel the agent's turn, end a running acquisition, and drop every schedule
        and the open request.

        A running acquisition ends after the image being taken. A single move
        or image the driver has already started runs to its end: the ZMART
        vocabulary has no command to interrupt it, so the microscope's own
        controls stop it sooner. The schedules go too, or one could start the
        microscope again a moment after Stop was pressed, and so does the
        request, with any wait it had pending.
        """
        self.cancel.set()
        self.scheduler.clear()
        self.requests.end("stopped")

    def forget(self) -> None:
        """Forget the session's side: plans, go-aheads, schedules, requests, frames, eyes.

        Clear context and choosing another microscope call this (through
        ``Conversation.clear``); the connection stays.
        """
        self.plans.clear()
        self.planned_in.clear()
        self.go_ahead_asked.clear()
        self.scheduler.clear()
        self.requests.end("cleared")
        self.requests = Requests(self.now)
        self.frames.clear()
        self.seen = None
        self.eyes.reset()


FOLDED_ERROR = re.compile(r"^([A-Za-z_]\w*(?:Error|Exception)): (.*)$", re.DOTALL)


def folded_error(answer: Any) -> Exception | None:
    """The error a controller answer folded in, or None for a success or a soft answer.

    A ValueError comes back as a ValueError, with the driver's message; any
    other error as a RuntimeError, which keeps the error's name unless it was
    one already.
    """
    if (
        not isinstance(answer, dict)
        or answer.get("success")
        or not isinstance(answer.get("content"), str)
    ):
        return None
    match = FOLDED_ERROR.match(answer["content"])
    if match is None:
        return None
    name, message = match.groups()
    if name == "ValueError":
        return ValueError(message)
    return RuntimeError(message if name == "RuntimeError" else f"{name}: {message}")


def learn(session: ZmartController) -> dict[str, Any]:
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
        low, high = reading.get("canvas") or (None, None)
        span = f"canvas {low:g} to {high:g} um" if None not in (low, high) else "canvas not given"
        # The motors of the axis: as get_actuators lists them, or else the ones
        # get_xyz reports a reading for.
        motors = ", ".join(str(m) for m in actuators.get(axis) or reading.get("actuators") or [])
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
