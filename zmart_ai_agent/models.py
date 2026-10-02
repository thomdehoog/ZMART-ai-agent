"""Which model the agent talks to, and how it is reached.

The chat window offers a few providers (``PROVIDERS`` in ``settings.py``).
Choosing one fills in a sensible model name (and, for a server you run
yourself, its address); the operator types the API key into the window. The
key stays in memory for the session only: it is never written to a file, a
log, or the conversation. An empty key field falls back to the environment
variable named in the preset, so `set GEMINI_API_KEY=...` before starting
still works.

Three kinds of connection exist:

- ``google``: Gemini, through Google's API.
- ``openai``: OpenAI's own API.
- ``openai-compatible``: any server that speaks the OpenAI chat API. That is
  Ollama, vLLM, LM Studio, a company gateway, or a model file this window serves
  itself (see ``local.py``). Such a server needs a base URL, and a key only if
  it asks for one.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .settings import MODEL_SETTINGS, PREFIXES, PROVIDERS


@dataclass
class Endpoint:
    """One model as chosen in the window. The key is held in memory only."""

    provider: str
    kind: str
    model: str
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    vision: bool = False  # may be shown a camera image (the "look" tool's side call)

    @classmethod
    def from_preset(
        cls,
        provider: str,
        model: str = "",
        api_key: str = "",
        base_url: str = "",
        vision: bool | None = None,
    ) -> Endpoint:
        """Fill the blanks from the provider preset.

        An empty model or base URL takes the preset's; an empty key takes the
        preset's environment variable, which may itself be unset.
        """
        preset = PROVIDERS[provider]
        key = api_key.strip()
        for variable in (preset.get("key_env"), preset.get("key_env_also")):
            if not key and variable:
                key = os.environ.get(variable, "")
        return cls(
            provider=provider,
            kind=preset["kind"],
            model=model.strip() or preset["model"],
            api_key=key,
            base_url=base_url.strip() or preset.get("base_url", ""),
            vision=bool(preset.get("vision", False)) if vision is None else vision,
        )

    @classmethod
    def from_name(cls, name: str) -> Endpoint:
        """An endpoint from a pydantic-ai style name such as ``google:gemini-3.5-flash-lite``.

        This is what the ``--model`` command line option takes. A name without a
        known provider prefix is sent to an OpenAI-style server.
        """
        prefix, _, model = name.partition(":")
        provider = PREFIXES.get(prefix.lower())
        if provider is None:
            return cls.from_preset("OpenAI-style server", model=name)
        return cls.from_preset(provider, model=model)

    @property
    def needs_key(self) -> bool:
        return self.kind != "openai-compatible"

    @property
    def key_variable(self) -> str | None:
        """The environment variable that could hold this endpoint's key, if any."""
        return PROVIDERS[self.provider].get("key_env")

    @property
    def name(self) -> str:
        """A short label for the window's status line and error messages."""
        return f"{self.provider}: {self.model}"

    @property
    def settings(self) -> dict[str, Any]:
        return MODEL_SETTINGS[self.kind]


def build_model(endpoint: Endpoint) -> Any:
    """The pydantic-ai model object for an endpoint.

    The provider libraries are imported here, not at the top, so the window starts
    even when only one of them is installed.
    """
    if endpoint.kind == "google":
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel(endpoint.model, provider=GoogleProvider(api_key=endpoint.api_key))
    # OpenAI itself, and any OpenAI-compatible server; a local server needs no key.
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(
        base_url=endpoint.base_url or None, api_key=endpoint.api_key or "not-needed"
    )
    return OpenAIChatModel(endpoint.model, provider=provider)


def missing_key_advice(endpoint: Endpoint) -> str:
    """One sentence on how to give the key, for a message to the operator."""
    if not endpoint.needs_key:
        return "This server refused the request; if it wants a key, type it in the Model panel."
    variable = endpoint.key_variable
    return (
        f"Type the {endpoint.provider} API key in the Model panel and press Use this model, "
        f"or set {variable} before starting."
    )
