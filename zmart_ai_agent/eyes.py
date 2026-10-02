"""The eyes: the vision model's own conversation for the session.

Every image the assistant looks at is a turn in this conversation, with the
time it was taken, the microscope's settings and the measured numbers, so the
eyes can compare the current image with earlier ones ("is it sharper than
before?", "has the sample moved since the first image?") and can be asked
about the images seen without taking a new one. The chat model itself never
carries an image; it gets the eyes' answer in words.

The newest VISION_FRAMES_KEPT images stay attached; older turns keep their
text (time, settings, numbers, and what the eyes said) and lose the picture.
Beyond VISION_TURNS_KEPT looks the oldest turns are dropped altogether, so the
cost of a look stays about one image however long the session runs.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any

import numpy as np
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.messages import ModelMessage, UserPromptPart

from .images import as_png
from .instructions import EYES_INSTRUCTIONS
from .settings import CLOCK_FORMAT, TEMPERATURE, VISION_FRAMES_KEPT, VISION_TURNS_KEPT


class Eyes:
    """The vision model with a memory of this session's images.

    ``model`` is a Pydantic AI model name or model object. ``look`` shows it a
    new image with a question; ``ask`` puts a question about the images already
    seen; ``reset`` forgets them all (Clear context does this).
    """

    def __init__(
        self,
        model: Any,
        frames_kept: int = VISION_FRAMES_KEPT,
        turns_kept: int = VISION_TURNS_KEPT,
    ) -> None:
        self.model = model
        self.frames_kept = frames_kept
        self.turns_kept = turns_kept
        self.frames = 0  # images seen this session
        self._history: list[ModelMessage] = []
        self._agent: Agent | None = None

    async def look(
        self, image: np.ndarray, question: str, stats: dict, context: dict | None = None
    ) -> str:
        """Show the eyes a new image and return their answer to the question."""
        number = self.frames + 1  # counted once the eyes have seen it
        text = f"Image {number}, {time.strftime(CLOCK_FORMAT)}."
        if context:
            text += f" Microscope: {json.dumps(context, default=str)}"
        text += f"\nQuestion: {question}\nMeasured on the raw image: {json.dumps(stats)}"
        answer = await self._run([text, as_png(image)])
        self.frames = number
        return answer

    async def ask(self, question: str) -> str:
        """A question about the images seen so far, with no new image."""
        if self.frames == 0:
            return "No image has been looked at yet in this session; look first."
        return await self._run(f"No new image. Question about the images seen so far: {question}")

    def reset(self) -> None:
        """Forget every image; Clear context and a change of vision model call this."""
        self._history, self.frames = [], 0

    async def _run(self, prompt: Any) -> str:
        if self._agent is None:
            self._agent = Agent(
                self.model,
                instructions=EYES_INSTRUCTIONS.format(kept=self.frames_kept),
                model_settings={"temperature": TEMPERATURE},
            )
        result = await self._agent.run(prompt, message_history=self._history)
        kept = last_turns(result.all_messages(), self.turns_kept)
        self._history = detach_old_frames(kept, self.frames_kept)
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
    """The messages with the picture removed from every image turn but the last ``kept``.

    The text of such a turn (time, settings, numbers) and the eyes' answer stay,
    so a comparison with an older image rests on those.
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
    return dataclasses.replace(part, content=[*kept, "[image no longer attached]"])
