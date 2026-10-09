"""Moving the stage: the move_stage tool, and the checks around a move.

A move is checked against the canvas the driver reports (a position outside
it is certainly out of reach) before anything is sent; then the driver checks
again, against its own travel limits, and refuses a move beyond them. A move
of more than CONFIRM_XY_UM in x or y, or CONFIRM_Z_UM in z, measured from
where the stage was when the operator last wrote, is agreed in the chat
first. The answer reports the position read back from the microscope after
the move, which is what a ZMART ``set_xyz`` answers anyway.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-09
License: MIT
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import ModelRetry, RunContext

from .instructions import LIMIT_ADVICE
from .microscope import Microscope
from .settings import CONFIRM_XY_UM, CONFIRM_Z_UM
from .tooling import AXES, answered, guarded_tool, needs_go_ahead, refusal


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
        if (outside := outside_canvas(xyz[axis], axis, value)) is not None:
            return refusal(ctx, "limit", f"{outside} The stage did not move.", LIMIT_ADVICE)
    here = {axis: xyz[axis]["position"] for axis in AXES}
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


def outside_canvas(reading: dict[str, Any], axis: str, value: float) -> str | None:
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
    """set_xyz, and where the stage ended up. What the driver refuses is a limit refusal.

    A successful set_xyz already answers with the position read back from the
    microscope, in the same form as get_xyz, so no second reading is needed.
    After a failed move the stage is read once more, to say where it stands.
    """
    try:
        answer = ctx.deps.call(
            "set_xyz", position["x"], position["y"], position["z"], with_actuators=actuators
        )
    except ValueError as exc:
        message = f"the driver refused the move: {exc}. The stage did not move."
        return refusal(ctx, "limit", message, LIMIT_ADVICE)
    content = answer.get("content")
    if answer.get("success") and isinstance(content, dict):
        reached = {axis: content[axis]["position"] for axis in AXES}
    else:
        reached = ctx.deps.position()
    return {**answered(answer), "position": reached}
