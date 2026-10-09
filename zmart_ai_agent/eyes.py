"""The eyes: the vision model's own conversation for the session.

Every look is a turn in this conversation. It shows the eyes the frames the
look asks about, oldest first, each with its number, the time it was taken,
the microscope's position and settings and the measured numbers, and then
the question. So the eyes can compare the frames they are shown ("is frame 7
sharper than frame 3?") and can be asked about the frames seen before
without a new picture. The chat model itself never carries an image; it gets
the eyes' answer in words.

Once a turn is answered, it keeps its words (the frames' numbers, measures,
and what the eyes said) and loses the pictures, so a look costs the frames it
shows and no more; to compare with an older frame, a look names it and it is
shown again from the frame history. Beyond VISION_TURNS_KEPT looks the oldest
turns are dropped altogether, so the conversation stays small however long
the session runs.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.messages import ModelMessage, UserPromptPart

from .images import as_png
from .instructions import EYES_INSTRUCTIONS
from .schedules import hms
from .settings import TEMPERATURE, VISION_TURNS_KEPT


class Eyes:
    """The vision model with a memory of this session's looks.

    ``model`` is a Pydantic AI model name or model object. ``look`` shows it
    frames with a question; ``ask`` puts a question about the frames already
    seen; ``reset`` forgets them all (Clear context does this). ``clock`` is
    the agent's clock, so the times the eyes read match the frames'.
    """

    def __init__(
        self,
        model: Any,
        turns_kept: int = VISION_TURNS_KEPT,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.model = model
        self.turns_kept = turns_kept
        self.clock = clock
        self.frames = 0  # looks answered this session
        self._history: list[ModelMessage] = []
        self._agent: Agent | None = None

    async def look(
        self,
        pictures: list[tuple[str, np.ndarray, int]],
        question: str,
        context: dict | None = None,
    ) -> str:
        """Show the eyes frames and return their answer to the question.

        ``pictures`` holds, oldest first, what each frame is (its number, time
        and measures, in words), its pixels, and the binning to show it with
        (1 for a small copy the history kept, LOOK_BIN for a fresh image).
        """
        parts: list[Any] = [f"{hms(self.clock())}."]
        if context:
            parts[0] += f" Microscope now: {json.dumps(context, default=str)}"
        for text, image, bin in pictures:
            parts += [text, as_png(image, bin=bin)]
        parts.append(f"Question: {question}")
        answer = await self._run(parts)
        self.frames += 1
        return answer

    async def ask(self, question: str) -> str:
        """A question about the frames seen so far, with no new picture."""
        if self.frames == 0:
            return "No image has been looked at yet in this session; look first."
        return await self._run(f"No new image. Question about the frames seen so far: {question}")

    def reset(self) -> None:
        """Forget every look; Clear context and a change of vision model call this."""
        self._history, self.frames = [], 0

    async def _run(self, prompt: Any) -> str:
        if self._agent is None:
            self._agent = Agent(
                self.model,
                instructions=EYES_INSTRUCTIONS,
                model_settings={"temperature": TEMPERATURE},
            )
        result = await self._agent.run(prompt, message_history=self._history)
        kept = last_turns(result.all_messages(), self.turns_kept)
        self._history = detach_old_frames(kept, 0)
        return result.output


def last_turns(messages: list[ModelMessage], kept: int) -> list[ModelMessage]:
    """The messages of the last ``kept`` looks or questions, oldest ones dropped.

    A turn starts at a message the eyes were asked (a UserPromptPart), so a
    question is never separated from its answer.
    """
    starts = [
        i
        for i, message in enumerate(messages)
        if any(isinstance(part, UserPromptPart) for part in getattr(message, "parts", []))
    ]
    if len(starts) <= kept:
        return list(messages)
    return list(messages[starts[-kept] :])


def detach_old_frames(messages: list[ModelMessage], kept: int) -> list[ModelMessage]:
    """The messages with the pictures removed from every look but the last ``kept``.

    The words of such a turn (the frames' numbers, times, measures) and the
    eyes' answer stay, so a comparison with an older frame rests on those.
    """
    with_image = [
        i
        for i, message in enumerate(messages)
        if any(_carries_image(part) for part in getattr(message, "parts", []))
    ]
    to_strip = set(with_image[:-kept] if kept > 0 else with_image)
    out = []
    for i, message in enumerate(messages):
        if i in to_strip:
            parts = [_without_image(part) for part in message.parts]
            message = dataclasses.replace(message, parts=parts)
        out.append(message)
    return out


def _carries_image(part: Any) -> bool:
    return (
        isinstance(part, UserPromptPart)
        and isinstance(part.content, list)
        and any(isinstance(c, BinaryContent) for c in part.content)
    )


def _without_image(part: Any) -> Any:
    if not _carries_image(part):
        return part
    kept = [c for c in part.content if not isinstance(c, BinaryContent)]
    return dataclasses.replace(part, content=[*kept, "[pictures no longer attached]"])
