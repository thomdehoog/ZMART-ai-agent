"""The agent, assembled: the Pydantic AI ``Agent`` with its tools, and one conversation.

Built on Pydantic AI. The tools (``tools.py``) are what the model can ask the
microscope to do; the instructions are the generic part from
``instructions.py``, the same for every microscope, followed by the section
about the connected microscope, which ``microscope.py`` writes from its
driver's answers; the memory (``memory.py``) keeps a long conversation small;
the models (``models.py``) are the ways to reach a model. ``Conversation`` is one
conversation: a message in, the answer out.

    microscope = Microscope("mock")
    conversation = Conversation(microscope)
    print(conversation.send("Take a picture here and tell me what you see"))

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai import Agent, RunContext, capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelRequest

from . import models
from .instructions import INSTRUCTIONS
from .memory import compact, without_a_declined_challenge, without_state_block
from .microscope import Microscope
from .settings import DEFAULT_MODEL_SETTINGS, MODEL, TOOL_CALL_RETRIES
from .tooling import flat_view
from .tools import REPLY_GUARDS, TOOLS

agent = Agent(
    deps_type=Microscope,
    output_type=str,
    instructions=INSTRUCTIONS,
    retries=TOOL_CALL_RETRIES,
    defer_model_check=True,
)
for tool in TOOLS:
    agent.tool(sequential=True)(tool)  # one tool call at a time, so each result is seen
for guard in REPLY_GUARDS:
    agent.output_validator(guard)


@agent.instructions
def this_microscope(ctx: RunContext[Microscope]) -> str:
    """The section about the connected microscope, added after the generic instructions."""
    return ctx.deps.instrument_section()


class Conversation:
    """A conversation with the microscope agent. Not thread-safe: one turn at a time."""

    def __init__(
        self,
        microscope: Microscope,
        model: Any = MODEL,
        model_settings: dict[str, Any] | None = None,
    ) -> None:
        self.microscope = microscope
        self.model = model
        self.model_settings = DEFAULT_MODEL_SETTINGS if model_settings is None else model_settings
        self.endpoint: models.Endpoint | None = None  # what the window chose, if it did
        self.history: list[ModelMessage] = []
        self.last_turn: list[ModelMessage] = []  # the latest turn's messages, for traces

    def send(self, text: str, origin: str = "operator", request: int | None = None) -> str:
        """One message in, the agent's answer out.

        A message the operator typed (``origin`` "operator") starts a new
        request and a new turn of theirs: moves are measured from where the
        stage is now, and a question the agent asked in the turn before counts
        as answered by this message. A turn the machine wrote, a schedule that
        fell due ("scheduled") or a wait that is over ("continuation"), belongs
        to the request numbered ``request`` and does neither, so a repeating
        schedule cannot creep the stage along in small steps, and cannot stand
        in for the operator's go-ahead.
        """
        microscope = self.microscope
        microscope.cancel.clear()
        if origin == "operator":
            microscope.requests.typed(text)
        else:
            microscope.requests.machine(request)
        try:
            microscope.ensure_connected()  # the first message, or after a failed connect
            state = microscope.state(origin)
            if origin == "operator":
                microscope.anchor = state["position_um"]
            microscope.seen = flat_view(state)
        except Exception as exc:
            # No microscope chosen, or it does not answer: the model still gets the
            # message, so it can call check_setup and tell the operator what to do.
            state = {"microscope": f"not connected: {exc}"}
            microscope.anchor, microscope.seen = None, None
        if origin == "operator":
            microscope.turn += 1
        prompt = f"{text}\n\n<microscope_state>{json.dumps(state, default=str)}</microscope_state>"
        with capture_run_messages() as messages:
            try:
                result = agent.run_sync(
                    prompt,
                    message_history=self.history,
                    deps=microscope,
                    model=self.model,
                    model_settings=self.model_settings,
                )
            except Exception:
                # When the model call fails after tools already ran (for example an
                # overloaded API), keep what happened: the next message then carries
                # those tool results, and the conversation can go on.
                if messages and isinstance(messages[-1], ModelRequest):
                    self.last_turn = list(messages[len(self.history) :])
                    self.history = list(messages)
                raise
        self.last_turn = without_a_declined_challenge(result.new_messages())
        self.history = compact(without_a_declined_challenge(result.all_messages()))
        reply = without_state_block(result.output)
        microscope.requests.finish_turn(reply, tokens_of(result.usage))
        return reply

    def use(self, endpoint: models.Endpoint, vision: models.Endpoint | None = None) -> None:
        """Talk to another model from the next message on; the conversation is kept.

        ``vision`` names the model shown camera images; None means the same one.
        """
        self.endpoint = endpoint
        self.model = models.build_model(endpoint)
        self.model_settings = endpoint.settings
        seeing = vision or endpoint
        self.microscope.vision_model = self.model if vision is None else models.build_model(seeing)
        self.microscope.vision = seeing.vision

    def clear(self) -> None:
        """Forget the conversation; the next message starts a new one.

        The eyes forget their looks, the frames go, and the schedules and the
        open request are cancelled.
        """
        self.history, self.last_turn = [], []
        self.microscope.forget()


def tokens_of(usage: Any) -> int:
    """The tokens a turn took, in and out, as the provider counted them."""
    if callable(usage):  # a method in some versions of Pydantic AI, a value in others
        usage = usage()
    return int(getattr(usage, "input_tokens", 0) or 0) + int(
        getattr(usage, "output_tokens", 0) or 0
    )
