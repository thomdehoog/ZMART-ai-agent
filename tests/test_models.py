"""Choosing a model: presets, keys from the window or the environment, and model objects.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

import pytest

pytest.importorskip("pydantic_ai")

from zmart_ai_agent import local, models
from zmart_ai_agent.models import Endpoint


def test_a_preset_fills_the_blanks(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    endpoint = Endpoint.from_preset("Gemini")
    assert endpoint.kind == "google" and endpoint.model == "gemini-3.5-flash-lite"
    assert endpoint.api_key == "" and endpoint.needs_key and endpoint.vision
    assert endpoint.name == "Gemini: gemini-3.5-flash-lite"


def test_a_typed_key_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert Endpoint.from_preset("Gemini").api_key == "from-env"
    assert Endpoint.from_preset("Gemini", api_key=" typed ").api_key == "typed"


def test_the_older_google_variable_still_counts(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "older-name")
    assert Endpoint.from_preset("Gemini").api_key == "older-name"


def test_a_server_needs_no_key_and_keeps_its_address():
    endpoint = Endpoint.from_preset("OpenAI-style server", base_url=" http://box:8000/v1 ")
    assert not endpoint.needs_key and endpoint.base_url == "http://box:8000/v1"
    assert not endpoint.vision
    assert Endpoint.from_preset("OpenAI-style server", vision=True).vision


def test_a_command_line_name_maps_to_a_preset(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    endpoint = Endpoint.from_name("openai:gpt-5-mini")
    assert endpoint.provider == "OpenAI" and endpoint.model == "gpt-5-mini"
    assert Endpoint.from_name("google-gla:gemini-3.5-flash-lite").provider == "Gemini"
    other = Endpoint.from_name("gemma4:31b")  # no known prefix: a server of one's own
    assert other.kind == "openai-compatible" and other.model == "gemma4:31b"


def test_each_kind_builds_its_model_object():
    google = models.build_model(Endpoint.from_preset("Gemini", api_key="k"))
    assert type(google).__name__ == "GoogleModel" and google.model_name == "gemini-3.5-flash-lite"
    server = models.build_model(Endpoint.from_preset("OpenAI-style server"))
    assert type(server).__name__ == "OpenAIChatModel" and server.model_name == "gemma4:31b"


def test_the_settings_follow_the_kind():
    assert Endpoint.from_preset("Gemini").settings["temperature"] == 0.0
    assert "parallel_tool_calls" not in Endpoint.from_preset("OpenAI-style server").settings


def test_the_advice_names_the_variable():
    advice = models.missing_key_advice(Endpoint.from_preset("OpenAI", api_key="x"))
    assert "OPENAI_API_KEY" in advice and "Model panel" in advice
    assert "server" in models.missing_key_advice(Endpoint.from_preset("OpenAI-style server"))


# -- model files on this computer -----------------------------------------------------------


def test_model_files_are_listed_without_their_projectors(tmp_path):
    for name in (
        "gemma-4-12b-it-Q4.gguf",
        "mmproj-gemma-4-12b-F16.gguf",
        "notes.txt",
        "qwen3.5-8b.gguf",
    ):
        (tmp_path / name).write_bytes(b"")
    assert local.list_models(tmp_path) == ["gemma-4-12b-it-Q4.gguf", "qwen3.5-8b.gguf"]
    assert local.list_models(tmp_path / "missing") == []


def test_a_projector_is_matched_by_family(tmp_path):
    for name in ("gemma-4-12b-it-Q4.gguf", "mmproj-gemma-4-12b-F16.gguf", "mmproj-qwen3.5-8b.gguf"):
        (tmp_path / name).write_bytes(b"")
    assert local.projector_for(tmp_path, "gemma-4-12b-it-Q4.gguf").endswith(
        "mmproj-gemma-4-12b-F16.gguf"
    )
    assert local.projector_for(tmp_path, "llama-3-8b.gguf") is None


def test_the_server_command_names_the_file_and_the_context(monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    argv = local.server_command("C:/models/gemma-4.gguf", 5005, projector="C:/models/mmproj.gguf")
    assert "--model" in argv and "C:/models/gemma-4.gguf" in argv and "5005" in argv
    assert argv[argv.index("--n_ctx") + 1] == str(local.CONTEXT_TOKENS)
    assert argv[argv.index("--clip_model_path") + 1] == "C:/models/mmproj.gguf"


def test_a_missing_runtime_says_how_to_install_it(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # makes the import fail
    with pytest.raises(RuntimeError, match="llama-cpp-python"):
        local.server_command("x.gguf", 1)


def test_a_server_child_is_started_and_stopped(tmp_path):
    import sys

    def command(model_path, port, projector=None, context_tokens=None):
        return [sys.executable, "-c", "import time; time.sleep(60)"]

    server = local.LocalModelServer(str(tmp_path / "m.gguf"), command=command)
    assert server.model == "m" and server.base_url.startswith("http://127.0.0.1:")
    server.start()
    try:
        assert server.ready() is False  # the child sleeps; nothing listens on the port
    finally:
        server.stop()
    assert server.ready() is False
