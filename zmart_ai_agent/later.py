"""What happens later: the schedule, cancel_schedule and wait tools.

A schedule carries an instruction out later, as if the operator typed it then
(``schedules.py``). A wait ends the current turn and lets the request go on
by itself once the time is up (``requests.py``). Both are kept by the
microscope and fired by the window's clock, never by the model, which cannot
keep time.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-09
License: MIT
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext

from .instructions import FAILURE_ADVICE, OPTIONS_ADVICE, WAIT_NOTE
from .microscope import Microscope
from .tooling import guarded_tool, refusal


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
    request = ctx.deps.requests.current
    try:
        added = ctx.deps.scheduler.add(
            name,
            instruction,
            every_seconds,
            in_seconds,
            at,
            request=request.number if request is not None else None,
        )
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


@guarded_tool
def wait(ctx: RunContext[Microscope], seconds: int) -> dict[str, Any]:
    """End this turn and let the request go on by itself after a pause: for a step
    that needs time to pass first ("let it settle for two minutes, then look").
    The request continues in a new turn that says how long was waited. After
    calling this, reply with one short sentence and call no more tools.

    Args:
        seconds: how long to wait; two minutes is 120.
    """
    try:
        pending = ctx.deps.requests.wait(seconds)
    except ValueError as exc:
        return refusal(ctx, "invalid", str(exc), FAILURE_ADVICE)
    return {"waiting": {"seconds": pending["seconds"]}, "note": WAIT_NOTE}
