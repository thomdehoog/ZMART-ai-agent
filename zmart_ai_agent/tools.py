"""The tools: everything the model can ask the microscope to do, one function each.

Each tool is a few ZMART Controller commands. It checks what it is asked
against what the driver itself reports (the canvas from ``get_xyz``,
the setting names from ``get_state``, the acquisition settings and routines it lists)
before acting, and answers with data the model can read: the result, or an
"error" with what was refused, why, and what to do next (the advice in
``instructions.py``). What the driver refuses (a ValueError) comes back as a
refusal too, in the driver's own words, and a failure at the microscope (a
RuntimeError) as a failure.

The big steps, starting an acquisition, running a routine and moving the
stage far, are not carried out in the turn the model first asks for them: the
tool answers that the operator's go-ahead is needed (``needs_go_ahead``), and
the step runs only in the next turn, after the operator has replied. That
rule is in this code, not in the model's instructions.

``TOOLS`` lists them; ``agent.py`` registers them on the Agent. This is the
place to look up or add a tool. At the end are two guards on the model's reply
itself (``REPLY_GUARDS``): an empty reply, and a reply that claims to have done
something in a turn that called no tool, each go back to the model once.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import json
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import zmart_controller
from pydantic import BaseModel
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.messages import ToolCallPart

from .images import image_statistics, read_saved, saved_files
from .instructions import (
    CALLED_NOTHING_CHALLENGE,
    CANCELLED_ADVICE,
    CHOOSE_STEPS,
    CONNECT_STEPS,
    EMPTY_REPLY_CHALLENGE,
    EMPTY_REPLY_FALLBACK,
    FAILURE_ADVICE,
    GO_AHEAD_ADVICE,
    LAST_IMAGE_QUESTION,
    LIMIT_ADVICE,
    NO_FOCUS_ADVICE,
    OPTIONS_ADVICE,
    START_ADVICE,
    UNCONFIRMED_ADVICE,
)
from .memory import _is_operator_turn
from .microscope import Microscope
from .plans import AcquisitionPlan, PositionSpec, count_acquisitions, describe, steps
from .settings import (
    CONFIRM_XY_UM,
    CONFIRM_Z_UM,
    FILES_LISTED,
    FOCUS_WORD,
    LOOK_LABEL,
    SOURCE_LINES,
    SOURCE_MATCHES,
    WAIT_STEP_S,
)

AXES = ("x", "y", "z")


def refusal(
    ctx: RunContext[Microscope], code: str, message: str, advice: str, **details: Any
) -> dict[str, Any]:
    """A refused action, as data the model reads, with what to do about it.

    Limit breaches and invalid values also go to the window's warning banner,
    so the operator sees them whatever the model says.
    """
    if code in ("limit", "invalid"):
        ctx.deps.on_warning(message)
    return {"error": {"code": code, "message": message, **details, "advice": advice}}


def answered(answer: dict[str, Any], **more: Any) -> dict[str, Any]:
    """A driver's answer for the model; success false carries the advice to say so."""
    result = {**more, "success": bool(answer.get("success")), "content": answer.get("content")}
    if not result["success"]:
        result["advice"] = UNCONFIRMED_ADVICE
    return result


def needs_go_ahead(ctx: RunContext[Microscope], key: str, summary: str) -> dict | None:
    """None if the operator has had the chance to agree to this action; else the
    answer that tells the agent to ask first.

    The first request is only noted. The same request in the operator's next
    turn (after they read the question and replied) goes ahead. Whether the
    reply was a yes is for the model to read; that the operator saw the
    question before anything happened is guaranteed here.
    """
    if ctx.deps.go_ahead_asked.pop(key, None) == ctx.deps.turn - 1:
        return None
    ctx.deps.go_ahead_asked[key] = ctx.deps.turn
    return {"status": "needs_go_ahead", "not_done_yet": summary, "advice": GO_AHEAD_ADVICE}


def guarded_tool(fn: Callable) -> Callable:
    """The checks every tool shares, around the tool itself.

    Before the tool runs: nothing runs after Cancel, and the window hears of
    each call as it starts. Afterwards: what the driver refuses (ValueError)
    becomes a refusal in the driver's own words, and any other error (the
    microscope fails, a full disk) a failure the agent can explain,
    instead of ending the turn with a crash. Pydantic AI's ModelRetry, which
    hands a malformed call back to the model, passes through unchanged.
    """

    def before(ctx: RunContext[Microscope], kwargs: dict) -> dict | None:
        if ctx.deps.cancel.is_set():
            return {"status": "cancelled", "advice": CANCELLED_ADVICE}
        args = {  # what the model asked for, without the arguments it left out
            k: v.model_dump(exclude_none=True) if isinstance(v, BaseModel) else v
            for k, v in kwargs.items()
            if v is not None
        }
        ctx.deps.on_tool(fn.__name__, args)
        return None

    def failed(ctx: RunContext[Microscope], exc: Exception) -> dict:
        if isinstance(exc, ValueError):
            return refusal(ctx, "invalid", f"the driver refused: {exc}", FAILURE_ADVICE)
        return {
            "error": {
                "code": "failed",
                "message": f"{type(exc).__name__}: {exc}",
                "advice": FAILURE_ADVICE,
            }
        }

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(ctx: RunContext[Microscope], *args: Any, **kwargs: Any) -> Any:
            if (stopped := before(ctx, kwargs)) is not None:
                return stopped
            try:
                return await fn(ctx, *args, **kwargs)
            except ModelRetry:
                raise
            except Exception as exc:
                return failed(ctx, exc)

        return async_wrapper

    @functools.wraps(fn)
    def wrapper(ctx: RunContext[Microscope], *args: Any, **kwargs: Any) -> Any:
        if (stopped := before(ctx, kwargs)) is not None:
            return stopped
        try:
            return fn(ctx, *args, **kwargs)
        except ModelRetry:
            raise
        except Exception as exc:
            return failed(ctx, exc)

    return wrapper


# -- the setup and reading ------------------------------------------------------------------


@guarded_tool
def check_setup(ctx: RunContext[Microscope]) -> dict[str, Any]:
    """Which driver is chosen, and whether it connects; with the steps for the
    operator when not.

    Call it when the microscope does not answer. It connects again and reads
    the microscope afresh; it moves nothing.
    """
    microscope = ctx.deps
    chosen = microscope.name if microscope.driver is not None else None
    result: dict[str, Any] = {"driver": chosen}
    if chosen is None:
        return {**result, "connected": False, "steps_for_the_operator": CHOOSE_STEPS}
    try:
        microscope.connect()
    except Exception:
        return {
            **result,
            "connected": False,
            "error": microscope.connect_error,
            "steps_for_the_operator": CONNECT_STEPS,
        }
    return {
        **result,
        "connected": True,
        "description": microscope.has_description,
        "output_root": microscope.learned["info"].get("output_root"),
    }


@guarded_tool
def get_status(ctx: RunContext[Microscope]) -> dict[str, Any]:
    """Everything the microscope reports now: each axis (position in um, motor,
    canvas) and the state (changeable settings and the read-only report)."""
    return {"position": ctx.deps.read("get_xyz"), "state": ctx.deps.read("get_state")}


# -- moving ---------------------------------------------------------------------------------


@guarded_tool
def move_stage(
    ctx: RunContext[Microscope],
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    actuators: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Move the stage to an absolute position, in micrometres in the microscope's frame.

    Args:
        x: x in um; leave out to keep.
        y: y in um; leave out to keep.
        z: z in um; leave out to keep.
        actuators: the motor per axis, by the names the microscope lists, e.g.
            {"z": "piezo"}; leave out for the usual one.
    """
    target = {axis: v for axis, v in (("x", x), ("y", y), ("z", z)) if v is not None}
    if not target:
        raise ModelRetry("Give at least one of x, y, z.")
    xyz = ctx.deps.read("get_xyz")
    for axis, value in target.items():
        if (outside := _outside(xyz[axis], axis, value)) is not None:
            return refusal(ctx, "limit", f"{outside} The stage did not move.", LIMIT_ADVICE)
    here = {axis: xyz[axis]["value"] for axis in AXES}
    anchor = ctx.deps.anchor or here
    xy_step = max(abs(target.get(a, here[a]) - anchor[a]) for a in ("x", "y"))
    z_step = abs(target.get("z", here["z"]) - anchor["z"])
    if xy_step <= CONFIRM_XY_UM and z_step <= CONFIRM_Z_UM:
        return _move(ctx, {**here, **target}, actuators)
    where = ", ".join(f"{a} = {v:g} um" for a, v in target.items())
    summary = (
        f"move the stage to {where}: {xy_step:.0f} um in XY and {z_step:.0f} um in Z "
        "from where it was when the operator last wrote"
    )
    key = f"move {sorted(target.items())} {sorted((actuators or {}).items())}"
    if (question := needs_go_ahead(ctx, key, summary)) is not None:
        return question
    moved = _move(ctx, {**here, **target}, actuators)
    if "position" in moved:
        ctx.deps.anchor = moved["position"]  # the operator agreed to this position
    return moved


def _outside(reading: dict[str, Any], axis: str, value: float) -> str | None:
    """Why ``value`` is outside an axis's canvas as get_xyz reports it, or None.

    The canvas is everywhere a picture can show, a little wider than the
    stage's travel, so a position outside it is certainly out of reach. A
    position inside it can still be beyond the travel: the driver keeps the
    travel limits and refuses such a move itself.
    """
    low, high = reading.get("canvas") or (None, None)
    if low is None or high is None or low <= value <= high:
        return None
    return (
        f"{axis} = {value:g} um is outside the canvas [{low:g}, {high:g}] um, "
        "everywhere a picture can show, so the stage cannot go there."
    )


def _move(ctx: RunContext[Microscope], position: dict[str, float], actuators) -> dict[str, Any]:
    """set_xyz, and the position read back. What the driver refuses is a limit refusal."""
    try:
        answer = ctx.deps.call(
            "set_xyz", position["x"], position["y"], position["z"], with_actuators=actuators
        )
    except ValueError as exc:
        message = f"the driver refused the move: {exc}. The stage did not move."
        return refusal(ctx, "limit", message, LIMIT_ADVICE)
    content = answer.get("content") or {}
    return {
        **answered(answer),
        "position": ctx.deps.position(),
        "actuators": content.get("actuators") if isinstance(content, dict) else None,
    }


# -- settings, focus and routines -------------------------------------------------------------


@guarded_tool
def set_microscope(ctx: RunContext[Microscope], settings: dict[str, Any]) -> dict[str, Any]:
    """Change settings, by the names the microscope lists as changeable. Leave out
    what should stay as it is.

    Args:
        settings: the new values by setting name, e.g. {"exposure_ms": 50}.
    """
    if not settings:
        raise ModelRetry("Give at least one setting.")
    changeable = ctx.deps.read("get_state").get("changeable") or {}
    unknown = [name for name in settings if name not in changeable]
    if unknown:  # checked first; then nothing changes
        names = ", ".join(repr(name) for name in unknown)
        message = f"{names} is not a setting this microscope can change"
        return refusal(ctx, "invalid", message, OPTIONS_ADVICE, configured_options=list(changeable))
    return answered(ctx.deps.call("set_state", {"changeable": dict(settings)}))


@guarded_tool
def focus(
    ctx: RunContext[Microscope],
    procedure: str | None = None,
    entries: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Find focus with the microscope's own focus routine (a listed routine whose
    name holds "focus"). Runs at once, without asking.

    Args:
        procedure: which focus routine, when the microscope lists several.
        entries: extra entries the routine takes, as its description names them,
            e.g. {"range_um": 20}.
    """
    procedures = ctx.deps.read("get_procedures")
    focusing = [name for name in procedures if FOCUS_WORD in name.lower()]
    if not focusing:
        message = "this microscope lists no focus procedure, so the agent cannot focus it"
        return refusal(
            ctx, "not_available", message, NO_FOCUS_ADVICE, configured_options=list(procedures)
        )
    if procedure is None:
        exact = [name for name in focusing if name.lower() == f"auto{FOCUS_WORD}"]
        procedure = (exact or focusing)[0] if len(focusing) == 1 or exact else None
        if procedure is None:
            message = f"this microscope offers several ways to focus: {', '.join(focusing)}"
            return refusal(ctx, "invalid", message, OPTIONS_ADVICE, configured_options=focusing)
    elif procedure not in focusing:
        message = f"{procedure!r} is not a focus procedure of this microscope"
        return refusal(ctx, "invalid", message, OPTIONS_ADVICE, configured_options=focusing)
    before = ctx.deps.position()["z"]
    answer = ctx.deps.call("run_procedure", {**(entries or {}), "name": procedure})
    after = ctx.deps.position()["z"]
    return answered(answer, procedure=procedure, z_before_um=before, z_after_um=after)


@guarded_tool
def run_procedure(
    ctx: RunContext[Microscope], name: str, entries: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run one of the routines the microscope lists. What a routine does is the
    microscope's to say, so it is agreed with the operator first.

    Args:
        name: the routine, as listed.
        entries: extra entries the routine takes, as its description names them.
    """
    procedures = ctx.deps.read("get_procedures")
    if name not in procedures:
        message = f"{name!r} is not a routine this microscope offers"
        return refusal(ctx, "invalid", message, OPTIONS_ADVICE, configured_options=list(procedures))
    spec = procedures[name]
    described = spec.get("description") if isinstance(spec, dict) else None
    with_entries = f" with {json.dumps(entries)}" if entries else ""
    summary = f"run the routine {name!r}{with_entries}" + (f": {described}" if described else "")
    key = f"procedure {name} {json.dumps(entries or {}, sort_keys=True)}"
    if (question := needs_go_ahead(ctx, key, summary)) is not None:
        return question
    answer = ctx.deps.call("run_procedure", {**(entries or {}), "name": name})
    with contextlib.suppress(Exception):  # the operator agreed to where the routine leaves it
        ctx.deps.anchor = ctx.deps.position()
    return answered(answer)


# -- looking --------------------------------------------------------------------------------


@guarded_tool
async def look(ctx: RunContext[Microscope], question: str) -> dict[str, Any]:
    """Acquire one image with the current settings, here, and answer a question about it.

    The answer comes from the eyes, which have seen every image of this session:
    ask them to compare with an earlier image when that is the question.

    Args:
        question: what to find out, e.g. "what do you see?", "is it in focus?",
            "is it sharper than the image before?".
    """
    label = f"{LOOK_LABEL}_{datetime.now():%Y%m%d_%H%M%S}"
    answer = await asyncio.to_thread(ctx.deps.call, "acquire", position_label=label)
    if not answer.get("success"):
        return answered(answer)
    files = saved_files(answer.get("content"))
    try:
        image = read_saved(files)
    except ValueError as exc:  # saved, but in a form the agent cannot read
        error = {"code": "unreadable", "message": str(exc), "advice": FAILURE_ADVICE}
        return {"files": files, "error": error}
    stats = image_statistics(image)
    ctx.deps.on_image(image, question)
    if not ctx.deps.vision:
        return {
            "statistics": stats,
            "files": files,
            "note": "the model in use cannot see images; judge from the numbers",
        }
    # A separate conversation: the image never enters the chat history, which
    # keeps long conversations small; the eyes remember it instead.
    eyes = ctx.deps.eyes
    answer_text = await eyes.look(image, question, stats, _image_context(ctx.deps))
    return {"answer": answer_text, "statistics": stats, "images_seen": eyes.frames, "files": files}


@guarded_tool
async def ask_eyes(ctx: RunContext[Microscope], question: str) -> dict[str, Any]:
    """Ask the eyes about the images already seen in this session, without taking
    a new image: "has the sample moved since the first image?", "which image
    was sharpest?".

    Args:
        question: what to compare or recall across the images seen.
    """
    if not ctx.deps.vision:
        return {"note": "the model in use cannot see images; nothing was looked at"}
    eyes = ctx.deps.eyes
    return {"answer": await eyes.ask(question), "images_seen": eyes.frames}


def _image_context(microscope: Microscope) -> dict[str, Any]:
    """Where the picture was taken, for the eyes' record of it; empty if the read fails."""
    try:
        return microscope.where()
    except (RuntimeError, OSError, ValueError):  # the microscope stopped answering
        return {}


async def describe_image(
    microscope: Microscope, image: np.ndarray, question: str
) -> dict[str, Any]:
    """The measured numbers for an image and, when the model can see, its answer.

    It runs on the agent's own event loop: the vision model may be the very
    same client as the chat model, and a client must stay on one loop. A failing
    description is reported, not raised, so a finished acquisition is never
    turned into a failure by the describing afterwards.
    """
    stats = image_statistics(image)
    if not microscope.vision:
        return {"statistics": stats, "note": "the model in use cannot see images"}
    try:
        answer = await microscope.eyes.look(image, question, stats, _image_context(microscope))
    except Exception as exc:
        return {"statistics": stats, "vision_error": f"{type(exc).__name__}: {exc}"}
    return {"statistics": stats, "description": answer}


# -- planning and running an acquisition ------------------------------------------------------


@guarded_tool
def plan_acquisition(ctx: RunContext[Microscope], plan: AcquisitionPlan) -> dict[str, Any]:
    """Check a plan against the microscope without moving: the positions against
    the travel range, the channels' settings against the changeable settings,
    and the acquisition settings against the ones the microscope lists.

    Returns a plan id and a summary to tell the operator before starting it.
    """
    here = ctx.deps.position()
    positions = plan.positions or [PositionSpec(**here, name="here")]  # "here" is fixed now
    positions = [
        p if p.name else p.model_copy(update={"name": f"p{i + 1}"}) for i, p in enumerate(positions)
    ]
    plan = plan.model_copy(update={"positions": positions})
    if (problem := check_plan(ctx, plan)) is not None:
        return problem
    plan_id = f"{plan.name}-{len(ctx.deps.plans) + 1}"
    ctx.deps.plans[plan_id] = plan
    ctx.deps.planned_in[plan_id] = ctx.deps.turn
    return {
        "plan_id": plan_id,
        "acquisitions": count_acquisitions(plan),
        "summary": describe(plan, here),
        "plan": plan.model_dump(exclude_defaults=True),
    }


def check_plan(ctx: RunContext[Microscope], plan: AcquisitionPlan) -> dict | None:
    """The first thing in a plan the microscope would refuse, as a refusal; else None."""
    for kind, names in (
        ("position", [p.name for p in plan.positions]),
        ("channel", [c.name for c in plan.channels]),
    ):
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:  # the saved files are named by them
            message = f"each {kind} needs its own name; used more than once: {', '.join(repeated)}"
            return refusal(ctx, "invalid", message, FAILURE_ADVICE)
    xyz = ctx.deps.read("get_xyz")
    for p in plan.positions:
        for axis in AXES:
            if (outside := _outside(xyz[axis], axis, getattr(p, axis))) is not None:
                message = f"position {p.name}: {outside} Nothing moved."
                return refusal(ctx, "limit", message, LIMIT_ADVICE)
    changeable = ctx.deps.read("get_state").get("changeable") or {}
    for channel in plan.channels:
        unknown = [name for name in channel.settings if name not in changeable]
        if unknown:
            names = ", ".join(repr(n) for n in unknown)
            message = f"channel {channel.name}: {names} is not a setting this microscope can change"
            return refusal(
                ctx, "invalid", message, OPTIONS_ADVICE, configured_options=list(changeable)
            )
    menu = ctx.deps.read("get_acquisition_settings")
    for where, settings in [("the plan", plan.acquisition_settings)] + [
        (f"channel {c.name}", c.acquisition_settings) for c in plan.channels
    ]:
        for name, value in settings.items():
            if name not in menu:
                message = f"{where}: {name!r} is not an acquisition setting of this microscope"
                return refusal(
                    ctx, "invalid", message, OPTIONS_ADVICE, configured_options=list(menu)
                )
            allowed = menu[name].get("options") if isinstance(menu[name], dict) else None
            if isinstance(allowed, list) and value not in allowed:
                message = f"{where}: {value!r} is not an allowed value of {name!r}"
                return refusal(ctx, "invalid", message, OPTIONS_ADVICE, configured_options=allowed)
    return None


@guarded_tool
async def run_acquisition(ctx: RunContext[Microscope], plan_id: str) -> dict[str, Any]:
    """Run a plan made by plan_acquisition: for every time point, position and
    channel, move there, apply the channel's settings and acquire. The driver
    saves the images; the answer names the files.

    Args:
        plan_id: the id plan_acquisition returned.
    """
    plan = ctx.deps.plans.get(plan_id)
    if plan is None:
        message = f"there is no plan with id {plan_id!r}"
        return refusal(
            ctx, "invalid", message, OPTIONS_ADVICE, configured_options=list(ctx.deps.plans)
        )
    if (problem := check_plan(ctx, plan)) is not None:  # the microscope may have changed
        return problem
    # The run goes ahead only in the turn right after the plan (or this question)
    # was shown, so that the operator's reply to it is the go-ahead. A plan from
    # earlier in the conversation is asked about again.
    if ctx.deps.planned_in[plan_id] != ctx.deps.turn - 1:
        ctx.deps.planned_in[plan_id] = ctx.deps.turn
        summary = f"start acquisition {plan_id!r}: {describe(plan, ctx.deps.position())}"
        return {"status": "needs_go_ahead", "not_done_yet": summary, "advice": START_ADVICE}
    if ctx.deps.cancel.is_set():  # Stop was pressed while the run was being prepared
        return {"status": "cancelled", "advice": CANCELLED_ADVICE}

    run = Run(ctx.deps, plan, plan_id)
    # The run blocks for as long as the acquisition takes; a thread keeps the
    # agent's own event loop free, which the vision request below needs.
    await asyncio.to_thread(run.go)
    with contextlib.suppress(Exception):  # the microscope may be gone after an error
        ctx.deps.anchor = ctx.deps.position()
    result: dict[str, Any] = {
        "acquisitions": run.acquisitions,
        "finished": run.finished,
        "duration_s": round(run.seconds, 1),
        "files_saved": len(run.files),
        "files": run.files[:FILES_LISTED],
    }
    if len(run.files) > FILES_LISTED:
        result["more_files"] = len(run.files) - FILES_LISTED
    if run.unconfirmed:
        result["unconfirmed"] = run.unconfirmed
        result["advice"] = UNCONFIRMED_ADVICE
    if run.refused:
        result.update(refusal(ctx, "invalid", f"the driver refused: {run.refused}", FAILURE_ADVICE))
    elif run.failure:
        result["error"] = {"code": "failed", "message": run.failure, "advice": FAILURE_ADVICE}
    if run.last is not None:
        # The image the operator sees on the right: a few numbers, and a short
        # description from the vision model, so the reply can say what was imaged.
        result["last_image"] = await describe_image(ctx.deps, run.last, LAST_IMAGE_QUESTION)
    return result


class Run:
    """One run of a plan, on its own thread: what it did, and how it ended.

    It checks for Stop before every move and every acquisition, so Stop ends
    a run after the image being taken. Each saved image is read back and shown
    in the window. A refusal or failure ends the run; what was saved until
    then stays saved and is listed.
    """

    def __init__(self, microscope: Microscope, plan: AcquisitionPlan, plan_id: str):
        self.microscope, self.plan, self.plan_id = microscope, plan, plan_id
        self.acquisitions = 0
        self.files: list[str] = []
        self.unconfirmed: list[dict[str, Any]] = []
        self.last: np.ndarray | None = None  # the image left in the window when the run ends
        self.finished = "completed"
        self.refused: str | None = None
        self.failure: str | None = None
        self.seconds = 0.0

    def go(self) -> None:
        started = time.monotonic()
        try:
            self._steps(started)
        except ValueError as exc:  # the driver refused a move or a setting
            self.finished, self.refused = "failed", str(exc)
        except Exception as exc:
            self.finished, self.failure = "failed", f"{type(exc).__name__}: {exc}"
        self.seconds = time.monotonic() - started

    def _steps(self, started: float) -> None:
        microscope, plan = self.microscope, self.plan
        where = None
        for t, position, channel, label in steps(plan):
            if not self._wait_until(started + t * plan.interval_s):
                return
            if position is not where:
                microscope.call("set_xyz", position.x, position.y, position.z)
                where = position
            if channel is not None and channel.settings:
                answer = microscope.call("set_state", {"changeable": dict(channel.settings)})
                if not answer.get("success"):  # never image a channel in the wrong settings
                    content = json.dumps(answer.get("content"), default=str)
                    raise RuntimeError(f"channel {channel.name} could not be set: {content}")
            if self._stopped():
                return
            settings = {
                **plan.acquisition_settings,
                **(channel.acquisition_settings if channel else {}),
            }
            answer = microscope.call(
                "acquire", position_label=label, acquisition_settings=settings or None
            )
            if not answer.get("success"):
                self.unconfirmed.append({"label": label, "content": answer.get("content")})
                continue
            self.acquisitions += 1
            files = saved_files(answer.get("content"))
            self.files += files
            try:
                self.last = read_saved(files)
            except ValueError:
                continue  # saved in a form this cannot show; the files are still listed
            microscope.on_image(self.last, f"{self.plan_id}: {label}")

    def _stopped(self) -> bool:
        if self.microscope.cancel.is_set():
            self.finished = "stopped"
            return True
        return False

    def _wait_until(self, moment: float) -> bool:
        """Wait for a time point's start, checking for Stop; False when stopped."""
        while not self._stopped():
            if time.monotonic() >= moment:
                return True
            time.sleep(WAIT_STEP_S)
        return False


# -- schedules ---------------------------------------------------------------------------


@guarded_tool
def schedule(
    ctx: RunContext[Microscope],
    name: str,
    instruction: str,
    every_seconds: int | None = None,
    in_seconds: int | None = None,
    at: str | None = None,
) -> dict[str, Any]:
    """Have an instruction carried out later, as if the operator typed it then:
    every_seconds repeats it, in_seconds does it once after a delay, at does it
    once at a clock time. Exactly one of the three is given. Returns the
    schedule as set and every schedule now in place.

    Args:
        name: a short name, to cancel it by.
        instruction: what to do then, in the operator's words: "look and tell me
            whether anything changed".
        every_seconds: repeat this often; every three minutes is 180.
        in_seconds: once, this long from now; in ten minutes is 600.
        at: once, at this clock time, 24-hour "HH:MM".
    """
    try:
        added = ctx.deps.scheduler.add(name, instruction, every_seconds, in_seconds, at)
    except ValueError as exc:
        return refusal(ctx, "invalid", str(exc), FAILURE_ADVICE)
    return {"scheduled": added, "schedules": ctx.deps.scheduler.listing()}


@guarded_tool
def cancel_schedule(ctx: RunContext[Microscope], name: str) -> dict[str, Any]:
    """Cancel a schedule by its name, or every one with "all".

    Args:
        name: the schedule's name, or "all".
    """
    cancelled = ctx.deps.scheduler.cancel(name)
    if not cancelled:
        names = [item["name"] for item in ctx.deps.scheduler.listing()]
        message = f"no schedule named {name!r}"
        return refusal(ctx, "invalid", message, OPTIONS_ADVICE, configured_options=names)
    return {"cancelled": cancelled, "schedules": ctx.deps.scheduler.listing()}


# -- reading the source ------------------------------------------------------------------


@guarded_tool
def search_source(ctx: RunContext[Microscope], text: str) -> dict[str, Any]:
    """Search the source code of this agent, of the ZMART Controller and of
    this microscope's driver for a word or phrase, to explain how something works.

    Returns matching lines as "file:line: text". When nothing matches, returns
    the list of files that can be read instead.

    Args:
        text: the word or phrase to find, for example "def set_xyz" or
            "autofocus"; upper and lower case do not matter.
    """
    files = source_files(ctx.deps)
    matches = [
        f"{name}:{number}: {line.strip()[:160]}"
        for name, path in files.items()
        for number, line in enumerate(_lines(path), start=1)
        if text.lower() in line.lower()
    ]
    if not matches:
        return {"matches": [], "files": list(files)}
    return {"matches": matches[:SOURCE_MATCHES], "more": max(0, len(matches) - SOURCE_MATCHES)}


@guarded_tool
def read_source(
    ctx: RunContext[Microscope], file: str, start_line: int = 1, lines: int = 80
) -> dict[str, Any]:
    """Read part of a source file of this agent, the controller or the driver,
    with line numbers.

    Args:
        file: a file as search_source names it, for example "zmart_controller/session.py".
        start_line: the first line to read, counting from 1.
        lines: how many lines to read, at most 200.
    """
    files = source_files(ctx.deps)
    if file not in files:
        message = f"{file!r} is not a source file here"
        return refusal(ctx, "not_found", message, OPTIONS_ADVICE, configured_options=list(files))
    text = _lines(files[file])
    start = max(1, start_line)
    chunk = text[start - 1 : start - 1 + max(1, min(lines, SOURCE_LINES))]
    return {
        "file": file,
        "lines": f"{start} to {start + len(chunk) - 1} of {len(text)}",
        "text": "\n".join(f"{start + i}: {line}" for i, line in enumerate(chunk)),
    }


def source_roots(microscope: Microscope) -> dict[str, Path]:
    """The source the agent may read: itself, the controller, and the connected
    microscope's driver when the controller knows where it lives. Nothing else."""
    roots = {
        "zmart_ai_agent": Path(__file__).resolve().parent,
        "zmart_controller": Path(zmart_controller.__file__).resolve().parent,
    }
    driver = microscope.driver_folder()
    # A driver inside the controller, such as its mock, is already readable there.
    if driver is not None and not any(driver.is_relative_to(r) for r in roots.values()):
        roots[driver.name] = driver
    return roots


def source_files(microscope: Microscope) -> dict[str, Path]:
    """The files the agent may read, by name: "zmart_controller/session.py", ..."""
    return {
        f"{label}/{path.relative_to(root).as_posix()}": path
        for label, root in source_roots(microscope).items()
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


TOOLS = (
    check_setup,
    get_status,
    move_stage,
    set_microscope,
    focus,
    run_procedure,
    look,
    ask_eyes,
    plan_acquisition,
    run_acquisition,
    schedule,
    cancel_schedule,
    search_source,
    read_source,
)


# -- guards on the reply ----------------------------------------------------------------


def hand_back_an_empty_reply(ctx: RunContext[Microscope], output: str) -> str:
    """A reply with no letter or digit goes back to the model once.

    A second empty reply reaches the operator as a plain sentence rather than as,
    say, an underscore.
    """
    asked = ctx.deps.__dict__.setdefault("_empty_asked", set())
    if re.search(r"[^\W_]", output or ""):
        asked.discard(ctx.run_id)
        return output
    if ctx.run_id in asked:
        asked.discard(ctx.run_id)
        return EMPTY_REPLY_FALLBACK
    asked.add(ctx.run_id)
    raise ModelRetry(EMPTY_REPLY_CHALLENGE)


GUARD_WORD = re.compile(r"^\s*SAME\b[\s.:!-]*")  # the one word CALLED_NOTHING_CHALLENGE asks for


def challenge_a_reply_that_called_nothing(ctx: RunContext[Microscope], output: str) -> str:
    """A small model answers "stop" with "I have stopped the microscope." and no call.

    The one thing known without reading the reply is that the turn called nothing,
    so such a reply goes back to the model once with that fact. If it then calls
    a tool, the turn goes on and its new reply reports what happened. If it does
    not, the operator gets the first reply word for word: asked to repeat itself
    a small model writes something shorter and worse, so it is asked for one word
    instead. Costs one short request on a turn that sends no command. Off unless
    the microscope's ``challenge_no_tool`` is set (the window sets it).
    """
    if not ctx.deps.challenge_no_tool or ctx.deps.cancel.is_set():
        return output
    first = ctx.deps.__dict__.setdefault("_first_reply", {})
    starts = [i for i, m in enumerate(ctx.messages) if _is_operator_turn(m)]
    turn = ctx.messages[starts[-1] :] if starts else ctx.messages
    called = any(isinstance(part, ToolCallPart) for m in turn for part in getattr(m, "parts", []))
    if not called and ctx.run_id in first:
        return first.pop(ctx.run_id)  # challenged, and still nothing called: as it was
    first.pop(ctx.run_id, None)
    # "SAME" and the challenge are for this guard, never for the operator: a reply
    # that opens with them, or holds nothing else, is asked for again.
    reply = GUARD_WORD.sub("", output, count=1).strip()
    if not reply or CALLED_NOTHING_CHALLENGE[:40] in reply:
        raise ModelRetry(EMPTY_REPLY_CHALLENGE)
    if called:
        return reply
    first[ctx.run_id] = reply
    raise ModelRetry(CALLED_NOTHING_CHALLENGE)


REPLY_GUARDS = (hand_back_an_empty_reply, challenge_a_reply_that_called_nothing)
