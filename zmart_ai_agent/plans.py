"""What the agent can plan, in what order it runs, and the plan in plain sentences.

A plan is deliberately simple, because it has to mean the same on every
microscope: positions, channels and time points, with one ``acquire`` for
every combination. A channel is a short name and the settings to apply before
imaging it, by the microscope's own setting names (the ``changeable`` part of
``get_state``). Anything an acquisition itself can do, such as a z-stack, is
one of the driver's acquisition settings (``get_acquisition_settings``), given
once for the whole plan or per channel. The run goes time point by time point, position by position,
and channel by channel.

``AcquisitionPlan`` is the form the model fills in. ``steps`` lists the
acquisitions in the order they run, with the label each one is saved under,
and ``describe`` puts a plan into plain sentences for the operator.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field

from .settings import CONFIRM_XY_UM, CONFIRM_Z_UM, PLAN_MAX_POSITIONS, PLAN_MAX_TIME_POINTS

NAME = r"^[A-Za-z0-9_-]{1,40}$"  # a name that is safe in any file name


class PositionSpec(BaseModel):
    x: float = Field(description="x in um, in the microscope's frame")
    y: float = Field(description="y in um")
    z: float = Field(description="z in um")
    name: str | None = Field(
        None, pattern=NAME, description="optional label for the files, e.g. 'well_A1'"
    )


class ChannelSpec(BaseModel):
    name: str = Field(pattern=NAME, description="short name for the files, e.g. 'gfp'")
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="the settings to apply before imaging this channel, by the names in "
        "changeable, e.g. {'exposure_ms': 50}",
    )
    acquisition_settings: dict[str, Any] = Field(
        default_factory=dict,
        description="acquisition settings for this channel only, on top of the plan's",
    )


class AcquisitionPlan(BaseModel):
    """One acquisition: positions x channels, optionally repeated in time, one acquire each."""

    name: str = Field(pattern=NAME, description="short name for the files")
    positions: list[PositionSpec] = Field(
        default_factory=list,
        max_length=PLAN_MAX_POSITIONS,
        description="leave empty to image at the current position",
    )
    channels: list[ChannelSpec] = Field(
        default_factory=list, description="leave empty to image with the settings as they are"
    )
    acquisition_settings: dict[str, Any] = Field(
        default_factory=dict,
        description="acquisition settings for every acquire, by the names that "
        "get_acquisition_settings lists, e.g. a z-stack",
    )
    time_points: int = Field(1, ge=1, le=PLAN_MAX_TIME_POINTS)
    interval_s: float = Field(0.0, ge=0, description="time between the starts of time points")


def steps(plan: AcquisitionPlan) -> Iterator[tuple[int, PositionSpec, ChannelSpec | None, str]]:
    """Every acquisition of a plan whose positions are named, in the order it runs.

    Each step is (time point, position, channel or None, label). The label
    names the position, then the channel, then the time point when there are
    several, so the saved files sort the way the run went.
    """
    for t in range(plan.time_points):
        for position in plan.positions:
            for channel in plan.channels or [None]:
                parts = [position.name or "here"]
                if channel is not None:
                    parts.append(channel.name)
                if plan.time_points > 1:
                    parts.append(f"t{t:03d}")
                yield t, position, channel, "_".join(parts)


def count_acquisitions(plan: AcquisitionPlan) -> int:
    return plan.time_points * max(1, len(plan.positions)) * max(1, len(plan.channels))


def describe(plan: AcquisitionPlan, here: dict[str, float]) -> str:
    """A plan in plain sentences, including how far the stage will travel from
    ``here`` (the stage position now, per axis in um)."""
    channels = [
        f"{c.name} ({_values(c.settings)})" if c.settings else c.name for c in plan.channels
    ]
    with_own = [c.name for c in plan.channels if c.acquisition_settings]
    acquisition = _values(plan.acquisition_settings) or "as the microscope has them"
    if with_own:
        acquisition += f", with settings of their own for {', '.join(with_own)}"
    times = ""
    if plan.time_points > 1:
        times = f", {plan.time_points} time points {plan.interval_s:g} s apart"
    positions = []
    for i, p in enumerate(plan.positions):
        positions.append(f"{p.name or f'#{i + 1}'} at x {p.x:.0f}, y {p.y:.0f}, z {p.z:.0f} um")
    if len(positions) > 6:
        positions = [*positions[:6], f"and {len(positions) - 6} more"]
    xy = max((max(abs(p.x - here["x"]), abs(p.y - here["y"])) for p in plan.positions), default=0)
    z = max((abs(p.z - here["z"]) for p in plan.positions), default=0.0)
    travel = f"The stage travels up to {xy:.0f} um in XY and {z:.0f} um in Z from where it is now."
    if xy > CONFIRM_XY_UM or z > CONFIRM_Z_UM:
        travel += " This includes a long move."
    return (
        f"{count_acquisitions(plan)} acquisitions: channels "
        f"{', '.join(channels) or 'none, the settings as they are now'}; "
        f"acquisition settings {acquisition}{times}; "
        f"at {len(plan.positions)} position(s) ({'; '.join(positions)}). {travel}"
    )


def _values(values: dict[str, Any]) -> str:
    """{"laser_power": 5, "format": "ome-zarr"} as 'laser_power 5, format "ome-zarr"'."""
    return ", ".join(
        f"{key} {value:g}" if _is_number(value) else f"{key} {json.dumps(value)}"
        for key, value in values.items()
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
