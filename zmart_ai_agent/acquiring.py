"""Acquiring: checking a plan against the microscope, and running it.

``plan_acquisition`` checks a plan (``plans.py``) without moving anything:
the positions against the canvas, the channels' settings against the
changeable settings, and the acquisition settings against the ones the
microscope lists. ``run_acquisition`` runs a checked plan, but only in the
turn after the operator saw it, so that their reply is the go-ahead. ``Run``
is one run on its own thread: it checks for Stop before every move and every
acquisition, shows each saved image in the window, keeps it as a frame, and
reports what was saved however the run ended.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-09
License: MIT
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

import numpy as np
from pydantic_ai import RunContext

from .images import image_statistics, read_saved, saved_files, small_copy
from .instructions import (
    CANCELLED_ADVICE,
    FAILURE_ADVICE,
    LAST_IMAGE_QUESTION,
    LIMIT_ADVICE,
    OPTIONS_ADVICE,
    START_ADVICE,
    UNCONFIRMED_ADVICE,
)
from .looking import describe_image
from .microscope import Microscope
from .moving import outside_canvas
from .plans import AcquisitionPlan, PositionSpec, count_acquisitions, describe, steps
from .settings import FILES_LISTED, WAIT_STEP_S
from .tooling import AXES, guarded_tool, refusal


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
            if (outside := outside_canvas(xyz[axis], axis, getattr(p, axis))) is not None:
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
    if run.frames:
        result["frames"] = f"{run.frames[0]}-{run.frames[-1]}"
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
    a run after the image being taken. Each saved image is read back, shown
    in the window and kept as a frame. A refusal or failure ends the run;
    what was saved until then stays saved and is listed.
    """

    def __init__(self, microscope: Microscope, plan: AcquisitionPlan, plan_id: str):
        self.microscope, self.plan, self.plan_id = microscope, plan, plan_id
        self.acquisitions = 0
        self.files: list[str] = []
        self.frames: list[int] = []  # the numbers of the frames this run added
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
            with contextlib.suppress(Exception):  # a frame that cannot be kept is still saved
                where_taken = microscope.snapshot()
                entry = microscope.frames.add(
                    small_copy(self.last),
                    image_statistics(self.last),
                    f"run {self.plan_id}: {label}",
                    where_taken,
                )
                self.frames.append(entry["n"])

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
