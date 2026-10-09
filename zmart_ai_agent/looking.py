"""Looking at the sample: the look and ask_eyes tools, and describing an image.

``look`` asks the driver to acquire one image where the stage is, reads the
saved file back, keeps the image as a numbered frame (``frames.py``) with
its measured numbers, and asks the eyes (``eyes.py``) a question about it.
The chat model never sees the picture itself: it gets the numbers, the frame's
placing (where the signal sits and the move that would centre it) and the
eyes' answer in words. Naming earlier frames shows them to the eyes with the
new one, and the answer measures how the frames differ. ``ask_eyes`` asks
about the frames already seen without a new picture. ``calibrate`` measures
how the picture moves when the stage moves, for a driver whose camera
registration has not been measured, so that centring moves are right.
``describe_image`` is what a run calls for its last image.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-09
License: MIT
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import numpy as np
from pydantic_ai import RunContext

from .frames import FrameHistory, direction, field_um, flag, nominal_scale, shift
from .images import image_statistics, read_saved, saved_files, small_copy
from .instructions import FAILURE_ADVICE
from .microscope import Microscope
from .schedules import hms
from .settings import (
    CALIBRATE_CONFIDENCE_MIN,
    CALIBRATE_STEP_FRACTION,
    CALIBRATE_STEP_UM,
    LOOK_BIN,
    LOOK_LABEL,
)
from .tooling import answered, guarded_tool, needs_go_ahead, refusal

CANNOT_SEE = "the model in use cannot see images; judge from the numbers"


@guarded_tool
async def look(
    ctx: RunContext[Microscope],
    question: str,
    frames: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Acquire one image with the current settings, here, keep it as a numbered frame,
    and answer a question about it.

    The answer comes from the eyes. The frame's numbers say where the bright
    signal sits and the stage move that would centre it; to compare with
    earlier frames, name them in frames and the answer measures the shift
    and the change in sharpness between them.

    Args:
        question: what to find out, e.g. "what do you see?", "is it in focus?",
            "is it sharper than frame 3?".
        frames: earlier frames to show with the new one: "last 3", "1,7" or "3-10".
        label: a short name for the new frame, to find it again: "before".
    """
    stamp = f"{datetime.fromtimestamp(ctx.deps.now()):%Y%m%d_%H%M%S}"
    answer = await asyncio.to_thread(
        ctx.deps.call, "acquire", position_label=f"{LOOK_LABEL}_{stamp}"
    )
    if not answer.get("success"):
        return answered(answer)
    files = saved_files(answer.get("content"))
    try:
        image = read_saved(files)
    except ValueError as exc:  # saved, but in a form the agent cannot read
        error = {"code": "unreadable", "message": str(exc), "advice": FAILURE_ADVICE}
        return {"files": files, "error": error}
    stats = image_statistics(image)
    history = ctx.deps.frames
    fresh = history.add(small_copy(image), stats, "look", _where(ctx.deps), label)
    ctx.deps.on_image(image, question)
    try:
        shown = history.pick(frames) if frames is not None else []
    except ValueError as exc:
        return refusal(ctx, "invalid", str(exc), FAILURE_ADVICE)
    if all(f is not fresh for f in shown):
        shown.append(fresh)
    result: dict[str, Any] = {
        "frame": history.placing(fresh),
        "statistics": stats,
        "files": files,
    }
    if len(shown) > 1:
        result["frames"] = [history.brief(f) for f in shown]
        result["changes"] = history.compare(shown)
    if not ctx.deps.vision:
        result["note"] = CANNOT_SEE
        return result
    # A separate conversation: the pictures never enter the chat history, which
    # keeps long conversations small; the eyes see them instead.
    eyes = ctx.deps.eyes
    pictures = [
        (eyes_text(history, f), image if f is fresh else f["image"], LOOK_BIN if f is fresh else 1)
        for f in shown
    ]
    result["answer"] = await eyes.look(pictures, question, _where(ctx.deps, observed=False))
    result["looks"] = eyes.frames
    return result


@guarded_tool
async def ask_eyes(ctx: RunContext[Microscope], question: str) -> dict[str, Any]:
    """Ask the eyes about the frames already seen in this session, without taking
    a new image: "has the sample moved since the first frame?", "which frame
    was sharpest?".

    Args:
        question: what to compare or recall across the frames seen.
    """
    if not ctx.deps.vision:
        return {"note": "the model in use cannot see images; nothing was looked at"}
    eyes = ctx.deps.eyes
    return {"answer": await eyes.ask(question), "looks": eyes.frames}


@guarded_tool
async def calibrate(ctx: RunContext[Microscope], step_um: float | None = None) -> dict[str, Any]:
    """Measure how the picture moves when the stage moves, with the objective in use:
    an image here, a small move in x and an image, back, the same in y. The
    answer is kept for this microscope and objective, and from then on the
    frames' centre_move_um and the map use it instead of the frame rule. For a
    driver whose camera registration is not measured yet, or to check it.
    Needs a visible sample. Agreed with the operator first.

    Args:
        step_um: the test move in x and in y; a tenth of the field of view when
            left out.
    """
    microscope = ctx.deps
    where = _where(microscope)
    field = field_um(where.get("observed"), _image_shape(microscope))
    step = float(
        step_um or (round(CALIBRATE_STEP_FRACTION * field[0]) if field else CALIBRATE_STEP_UM)
    )
    summary = (
        f"calibrate: move the stage {step:g} um in x and back, then {step:g} um in y and back, "
        "taking an image at each stop"
    )
    if (question := needs_go_ahead(ctx, f"calibrate {step:g}", summary)) is not None:
        return question
    try:
        report = await asyncio.to_thread(_calibrate, microscope, step, field)
    except ValueError as exc:  # the driver refused a move, or the pictures told nothing
        return refusal(ctx, "invalid", str(exc), FAILURE_ADVICE)
    if "error" in report:
        return refusal(ctx, "invalid", report["error"], FAILURE_ADVICE)
    return report


def _calibrate(microscope: Microscope, step: float, field: tuple[float, float] | None) -> dict:
    """The measuring itself, on a thread: four images around three moves, and the
    picture shifts between them by phase correlation. Returns the report, or
    {"error": ...} when a shift could not be measured."""
    history = microscope.frames
    start = _snap(microscope, "calibrate")
    if (why := flag(start)) is not None:
        return {
            "error": f"frame {start['n']}: {why}; calibrate needs a visible sample that is not saturated"
        }
    here = start["position"]
    rows, report, used = [], {"step_um": step, "frames": [start["n"]]}, []
    for axis in ("x", "y"):
        moved = {**here, axis: here[axis] + step}
        microscope.call("set_xyz", moved["x"], moved["y"], moved["z"])
        after = _snap(microscope, "calibrate")
        microscope.call("set_xyz", here["x"], here["y"], here["z"])
        report["frames"].append(after["n"])
        found = shift(start["image"], after["image"])
        if found is None or found["confidence"] < CALIBRATE_CONFIDENCE_MIN:
            return {
                "error": f"the picture shift for a move in {axis} could not be measured (frames "
                f"{start['n']} and {after['n']}); the sample may have too little detail, or "
                "have left the field of view"
            }
        height, width = start["image"].shape
        right, down = found["right"] / width / step, found["down"] / height / step
        rows.append([right, down])
        axis_report = {"content_moves": direction(right, down), "confidence": found["confidence"]}
        if field is not None:  # 1 when the pixel size the driver reports is right
            moved_um = (found["right"] * field[0] / width, found["down"] * field[1] / height)
            axis_report["picture_um_per_stage_um"] = round(float(np.hypot(*moved_um)) / step, 3)
        report[axis] = axis_report
    nominal = nominal_scale(field)
    report["follows_the_frame_rule"] = nominal is not None and all(
        direction(*rows[i]) == direction(*nominal[i]) for i in range(2)
    )
    history.calibration.store(start["calibration"], rows, hms(microscope.now()), step)
    report["kept_as"] = start["calibration"]
    report["note"] = (
        "kept for this microscope and objective; the frames' centre_move_um and the map use it now"
    )
    return report


def _snap(microscope: Microscope, source: str) -> dict[str, Any]:
    """One image here, kept as a frame; raises ValueError when the driver did not deliver one."""
    stamp = f"{datetime.fromtimestamp(microscope.now()):%Y%m%d_%H%M%S_%f}"
    answer = microscope.call("acquire", position_label=f"{source}_{stamp}")
    if not answer.get("success"):
        raise ValueError(f"the acquisition could not be confirmed: {answer.get('content')}")
    image = read_saved(saved_files(answer.get("content")))
    microscope.on_image(image, source)
    return microscope.frames.add(
        small_copy(image), image_statistics(image), source, _where(microscope)
    )


def _image_shape(microscope: Microscope) -> list[int]:
    """The size of the pictures this microscope takes, from the last frame, or unknown."""
    frames = microscope.frames.frames
    if frames and frames[-1].get("image") is not None:
        return [int(frames[-1]["image"].shape[0]), int(frames[-1]["image"].shape[1])]
    return []


def eyes_text(history: FrameHistory, entry: dict[str, Any]) -> str:
    """What the eyes are told about a frame: its number, time, label, where and how it
    was taken, and the measured numbers."""
    brief = history.brief(entry)
    head = f"Frame {brief['n']}, {brief['time']}, {brief['source']}"
    if "label" in brief:
        head += f", labelled {brief['label']!r}"
    rest = {k: v for k, v in brief.items() if k not in ("n", "time", "source", "label")}
    return f"{head}. {json.dumps(rest, default=str, separators=(',', ':'))}"


def _where(microscope: Microscope, observed: bool = True) -> dict[str, Any]:
    """Where the picture was taken, for the frame's record; empty if the read fails."""
    try:
        return microscope.snapshot() if observed else microscope.where()
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
    history = microscope.frames
    entry = history.frames[-1] if history.frames else None
    text = eyes_text(history, entry) if entry is not None else f"Measured: {json.dumps(stats)}"
    try:
        answer = await microscope.eyes.look(
            [(text, image, LOOK_BIN)], question, _where(microscope, observed=False)
        )
    except Exception as exc:
        return {"statistics": stats, "vision_error": f"{type(exc).__name__}: {exc}"}
    return {"statistics": stats, "description": answer}
