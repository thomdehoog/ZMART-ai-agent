"""The tools: everything the model can ask the microscope to do, listed in one place.

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

The tools live in files named for what they do; ``TOOLS`` lists them, and
``agent.py`` registers them on the Agent. This is the place to look a tool
up. The checks every tool shares are in ``tooling.py``, and the two guards
on the model's reply itself in ``guards.py``.

| File | Tools |
|---|---|
| here | check_setup, get_status |
| moving.py | move_stage |
| adjusting.py | set_microscope, focus, run_procedure |
| looking.py | look, ask_eyes, calibrate |
| acquiring.py | plan_acquisition, run_acquisition |
| later.py | schedule, cancel_schedule, wait |
| source.py | search_source, read_source |

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import RunContext

from .acquiring import plan_acquisition, run_acquisition
from .adjusting import focus, run_procedure, set_microscope
from .guards import REPLY_GUARDS
from .instructions import CHOOSE_STEPS, CONNECT_STEPS
from .later import cancel_schedule, schedule, wait
from .looking import ask_eyes, calibrate, look
from .microscope import Microscope
from .moving import move_stage
from .source import read_source, search_source
from .tooling import guarded_tool


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


TOOLS = (
    check_setup,
    get_status,
    move_stage,
    set_microscope,
    focus,
    run_procedure,
    look,
    ask_eyes,
    calibrate,
    plan_acquisition,
    run_acquisition,
    schedule,
    cancel_schedule,
    wait,
    search_source,
    read_source,
)

__all__ = ["REPLY_GUARDS", "TOOLS"]
