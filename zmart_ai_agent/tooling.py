"""What every tool shares: the checks around a tool, its refusals, and the go-ahead rule.

A tool is a plain function that takes the microscope (through Pydantic AI's
``RunContext``) and what the model asked for, and answers with data the model
can read. The pieces here wrap every tool the same way:

- ``guarded_tool`` runs the checks before and after a tool: nothing runs
  after Cancel, nothing touches the microscope once the turn has asked to
  wait, the window hears of each call as it starts, and what the driver
  refuses or what fails at the microscope becomes a refusal or a failure the
  agent can explain, rather than a crash. A tool that changed something ends
  its answer with ``state_changed``: the position and settings that differ
  from what the model last saw.
- ``refusal`` and ``answered`` shape a refused action and a driver's answer.
- ``needs_go_ahead`` holds the rule that a big step (a long move, a routine,
  an acquisition) runs only in the turn after the operator saw the question.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-09
License: MIT
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic_ai import ModelRetry, RunContext

from .instructions import (
    CANCELLED_ADVICE,
    FAILURE_ADVICE,
    GO_AHEAD_ADVICE,
    UNCONFIRMED_ADVICE,
    WAITING_ADVICE,
)
from .microscope import Microscope

AXES = ("x", "y", "z")
# The tools that change the microscope, or take an image: refused once the turn has
# asked to wait, and answering with state_changed afterwards.
ACTING = {"move_stage", "set_microscope", "focus", "run_procedure", "run_acquisition", "calibrate"}
TOUCHING = ACTING | {"look"}


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
    question before anything happened is guaranteed here. A scheduled or
    continued turn is not the operator's, so it can never be the go-ahead.
    """
    if ctx.deps.go_ahead_asked.pop(key, None) == ctx.deps.turn - 1:
        return None
    ctx.deps.go_ahead_asked[key] = ctx.deps.turn
    return {"status": "needs_go_ahead", "not_done_yet": summary, "advice": GO_AHEAD_ADVICE}


def guarded_tool(fn: Callable) -> Callable:
    """The checks every tool shares, around the tool itself.

    Before the tool runs: nothing runs after Cancel, nothing touches the
    microscope once this turn has asked to wait, and the window hears of
    each call as it starts. Afterwards: what the driver refuses (ValueError)
    becomes a refusal in the driver's own words, and any other error (the
    microscope fails, a full disk) a failure the agent can explain,
    instead of ending the turn with a crash. Pydantic AI's ModelRetry, which
    hands a malformed call back to the model, passes through unchanged. A
    tool that acts ends its answer with what changed (``with_changes``).
    """
    name = fn.__name__

    def before(ctx: RunContext[Microscope], kwargs: dict) -> dict | None:
        if ctx.deps.cancel.is_set():
            return {"status": "cancelled", "advice": CANCELLED_ADVICE}
        args = {  # what the model asked for, without the arguments it left out
            k: v.model_dump(exclude_none=True) if isinstance(v, BaseModel) else v
            for k, v in kwargs.items()
            if v is not None
        }
        ctx.deps.on_tool(name, args)
        if name in TOUCHING and ctx.deps.requests.is_waiting():
            message = "this turn has asked to wait; the microscope is not touched again in it"
            return {"error": {"code": "waiting", "message": message, "advice": WAITING_ADVICE}}
        return None

    def after(ctx: RunContext[Microscope], result: Any) -> Any:
        return with_changes(ctx.deps, result) if name in ACTING else result

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
                return after(ctx, await fn(ctx, *args, **kwargs))
            except ModelRetry:
                raise
            except Exception as exc:
                return after(ctx, failed(ctx, exc))

        return async_wrapper

    @functools.wraps(fn)
    def wrapper(ctx: RunContext[Microscope], *args: Any, **kwargs: Any) -> Any:
        if (stopped := before(ctx, kwargs)) is not None:
            return stopped
        try:
            return after(ctx, fn(ctx, *args, **kwargs))
        except ModelRetry:
            raise
        except Exception as exc:
            return after(ctx, failed(ctx, exc))

    return wrapper


# -- what changed since the model last saw the microscope ---------------------------------


def flat_view(state: dict[str, Any] | None) -> dict[str, Any]:
    """The position and the settings as one flat dictionary: "position.x",
    "settings.exposure_ms", ... so that two readings compare key by key."""
    out: dict[str, Any] = {}
    for group, label in (("position_um", "position"), ("settings", "settings")):
        values = (state or {}).get(group)
        if isinstance(values, dict):
            out.update({f"{label}.{name}": value for name, value in values.items()})
    return out


def with_changes(microscope: Microscope, result: Any) -> Any:
    """The result with ``state_changed``: the position and settings that differ from
    what the model last saw (the turn's state reading, then each result since).

    A move's new position otherwise sits deep in its result, and what a
    routine or a run did to the microscope nowhere. A microscope that cannot
    be read leaves the result as it is.
    """
    if not isinstance(result, dict):
        return result
    try:
        now = flat_view(microscope.where())
    except Exception:
        return result
    seen, microscope.seen = microscope.seen, now
    if seen is None:
        return result
    changed = {key: value for key, value in now.items() if seen.get(key) != value}
    if changed:
        result["state_changed"] = changed
    return result
