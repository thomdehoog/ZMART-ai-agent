"""Adjusting the microscope: changing settings, focusing, and running its routines.

``set_microscope`` changes settings by the names the driver lists as
changeable, and nothing else: a name the microscope does not know is refused
before anything is applied, with the microscope's own names. ``focus`` runs
the microscope's own focus routine at once, since focusing is routine work;
``run_procedure`` runs any other routine, after the operator's go-ahead,
because what a routine does is the microscope's to say.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-09
License: MIT
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

from pydantic_ai import ModelRetry, RunContext

from .instructions import NO_FOCUS_ADVICE, OPTIONS_ADVICE
from .microscope import Microscope
from .settings import FOCUS_WORD
from .tooling import answered, guarded_tool, needs_go_ahead, refusal


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
