"""The agent's memory: the conversation made smaller now and then, and a
quoted state block taken out of a reply.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from .instructions import CALLED_NOTHING_CHALLENGE
from .settings import (
    HISTORY_COMPACT_AFTER,
    HISTORY_FULL_TURNS,
    HISTORY_KEEP_TURNS,
    HISTORY_RESULT_CHARS,
)

STATE_BLOCK = re.compile(
    r"\s*<microscope_state(_then)?>.*?(</microscope_state(_then)?>|\Z)", re.DOTALL
)


def without_state_block(reply: str) -> str:
    """The reply without a copy of the <microscope_state> block.

    Some models paste the state reading they were sent back into their answer,
    although told not to. The operator should not see it, so it is taken out
    here. The model's own copy in the history stays as it was, since editing
    an earlier answer would spoil the model's check on its earlier reasoning.
    """
    return STATE_BLOCK.sub("", reply).strip() or "(The agent gave no answer in words.)"


def without_a_declined_challenge(messages: list[ModelMessage]) -> list[ModelMessage]:
    """The turn without the reply guard's challenge, when the model only confirmed.

    When a turn that called no tool is challenged (tools.challenge_a_reply_that_called_nothing)
    and the model still calls nothing, the operator gets the first reply, and the
    challenge and the model's "SAME" are of no further use. Left in the history,
    they teach the model to open its next replies with "SAME", or to echo the
    challenge, and the guard then passes that on as a first reply. So the
    conversation ends at the first reply, as the operator saw it. Only the tail
    is cut; no earlier answer is edited. A challenge that led to a tool call stays.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, ModelRequest) and any(
            isinstance(part, RetryPromptPart) and CALLED_NOTHING_CHALLENGE in str(part.content)
            for part in message.parts
        ):
            acted = any(
                isinstance(part, ToolCallPart) for m in messages[index:] for part in m.parts
            )
            return messages if acted else messages[:index]
        if _is_operator_turn(message):
            break
    return messages


def compact(messages: list[ModelMessage]) -> list[ModelMessage]:
    """The conversation made smaller once it has grown long, else unchanged.

    Once there are more than HISTORY_COMPACT_AFTER operator turns, the oldest are
    forgotten so that HISTORY_KEEP_TURNS remain. In those, all but the newest
    HISTORY_FULL_TURNS keep a one-line reading of the microscope instead of the
    whole state block, and long tool results are cut short. What the model and
    the operator said, and which tools were called, stays.

    Why only now and then, and between turns: some models check that their
    earlier reasoning (their "thinking") was written for exactly the conversation
    they are sent back with. Rewriting old turns on every message would make that
    check fail each time. So the history only ever grows, except at these
    compaction points, and there the old reasoning is left out altogether,
    which the check allows.
    """
    starts = [i for i, m in enumerate(messages) if _is_operator_turn(m)]
    if len(starts) <= HISTORY_COMPACT_AFTER:
        return messages
    kept = messages[starts[-HISTORY_KEEP_TURNS] :]
    full_from = starts[-HISTORY_FULL_TURNS] - starts[-HISTORY_KEEP_TURNS]  # index in kept
    out: list[ModelMessage] = []
    for index, message in enumerate(kept):
        if isinstance(message, ModelResponse):
            parts = [p for p in message.parts if not isinstance(p, ThinkingPart)]
            message = dataclasses.replace(message, parts=parts or [TextPart("(no reply)")])
        elif index < full_from:
            message = dataclasses.replace(message, parts=[_shorten(p) for p in message.parts])
        out.append(message)
    return out


def _is_operator_turn(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and isinstance(message.parts[0], UserPromptPart)


def _shorten(part: Any) -> Any:
    """An older message part, made small: a one-line state reading, a cut tool result."""
    if isinstance(part, UserPromptPart) and isinstance(part.content, str):
        match = re.search(r"<microscope_state>(.*?)</microscope_state>", part.content, re.DOTALL)
        if match:
            state = json.loads(match.group(1))
            then = {"position_um": state.get("position_um"), "settings": state.get("settings")}
            reading = f"<microscope_state_then>{json.dumps(then)}</microscope_state_then>"
            content = part.content[: match.start()] + reading + part.content[match.end() :]
            return dataclasses.replace(part, content=content)
    if isinstance(part, ToolReturnPart):
        text = part.content if isinstance(part.content, str) else json.dumps(part.content)
        if len(text) > HISTORY_RESULT_CHARS:
            return dataclasses.replace(
                part, content=text[:HISTORY_RESULT_CHARS] + " ... (shortened in memory)"
            )
    return part
