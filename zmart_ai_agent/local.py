"""Serve a model file on this computer, so the agent works without the internet.

A ``.gguf`` model file chosen in the window is served by llama.cpp's
OpenAI-compatible server (the ``llama_cpp.server`` module of the
``llama-cpp-python`` package) as a child process that only listens on this
computer. The agent then talks to it exactly as it would to any other
OpenAI-style server, so nothing in the agent depends on the runtime. The child
lives while that model is in use: choosing another model or closing the window
stops it.

Install the runtime with ``pip install "llama-cpp-python[server]"`` (a build
with GPU support is much faster; see that package's documentation). Model files
come from Hugging Face, for example a Gemma or Qwen ``.gguf``; put them in the
models folder shown in the window. A model can also see images when a matching
projector file (its name contains ``mmproj``) sits next to it.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import atexit
import http.client
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from .settings import (
    BATCH_TOKENS,
    CONTEXT_TOKENS,
    FLASH_ATTENTION,
    MODEL_SUFFIXES,
    MODELS_FOLDER,
)

DEFAULT_FOLDER = Path.home() / MODELS_FOLDER
# A server you run yourself may keep a small window: Ollama loads a model with
# 4,096 tokens unless told otherwise and refuses every request here outright.
CONTEXT_TOO_SMALL_SIGNS = ("exceed_context_size", "exceeds the available context size")
CONTEXT_TOO_SMALL_HELP = (
    "The model server's context window is smaller than one request (about 6,000 tokens). "
    "Give it 16,384 or more: for Ollama, set OLLAMA_CONTEXT_LENGTH=16384 on the server, "
    "or make a copy of the model with PARAMETER num_ctx 16384."
)


def list_models(folder: str | Path) -> list[str]:
    """Model files in the folder by name; a missing folder lists nothing."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        p.name
        for p in folder.iterdir()
        if p.name.lower().endswith(MODEL_SUFFIXES) and "mmproj" not in p.name.lower()
    )


def projector_for(folder: str | Path, model_name: str) -> str | None:
    """The projector file that lets a local model see, or None.

    A projector belongs to one model family, so the file must carry the model's
    family name: the first two dash-separated parts of the file name ("gemma-4",
    "qwen3.5-8b"). Of several, the one sharing the longest start with the model's
    name wins. A projector of another family makes the server fail or see nonsense,
    so none is better than a wrong one.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return None
    stem = model_name.lower().rsplit(".", 1)[0]
    family = "-".join(stem.split("-")[:2])
    candidates = [
        p
        for p in folder.iterdir()
        if p.name.lower().endswith(MODEL_SUFFIXES)
        and "mmproj" in p.name.lower()
        and family in p.name.lower()
    ]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda p: len(os.path.commonprefix([stem, p.name.lower().replace("mmproj-", "")])),
    )
    return str(best)


def server_command(
    model_path: str, port: int, projector: str | None = None, context_tokens: int | None = None
) -> list[str]:
    """The command line of the child that serves ``model_path`` on ``port``.

    Raises with the install hint when the runtime is missing, so the window can
    say what to do instead of failing later.
    """
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        raise RuntimeError(
            'llama-cpp-python is not installed: pip install "llama-cpp-python[server]"'
        ) from None
    argv = [
        sys.executable,
        "-m",
        "llama_cpp.server",
        "--model",
        model_path,
        "--model_alias",
        model_name(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--n_gpu_layers",
        "-1",  # everything the GPU can take; CPU-only builds ignore it
        "--n_ctx",
        str(int(context_tokens or CONTEXT_TOKENS)),
        "--n_batch",
        str(BATCH_TOKENS),
        "--flash_attn",
        "true" if FLASH_ATTENTION else "false",
    ]
    if projector:
        argv += ["--clip_model_path", projector]
    return argv


def model_name(model_path: str) -> str:
    return Path(model_path).stem


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class LocalModelServer:
    """One server child for one model file.

    ``start()`` returns at once; ask ``ready()`` until the model has loaded (a
    large file takes tens of seconds), then use ``base_url`` and ``model``.
    """

    def __init__(
        self,
        model_path: str,
        command=server_command,
        projector: str | None = None,
        context_tokens: int | None = None,
    ) -> None:
        self.model_path = model_path
        self.projector = projector
        self.context_tokens = context_tokens
        self.model = model_name(model_path)
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self.log_path: str | None = None
        self._command = command
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        # Built first: a missing runtime raises here, before anything is spawned.
        argv = self._command(self.model_path, self.port, self.projector, self.context_tokens)
        handle, self.log_path = tempfile.mkstemp(prefix="zmart-ai-agent-model-", suffix=".log")
        with os.fdopen(handle, "wb") as log:  # the child inherits the file; ours can close
            try:
                self._process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            except OSError:
                os.unlink(self.log_path)
                raise
        atexit.register(self.stop)  # no orphaned child if the window ends without stop()

    def ready(self) -> bool:
        """True once the server answers. Raises when the child has already exited."""
        if self._process is None:
            return False
        if self._process.poll() is not None:
            raise RuntimeError(
                f"the model server exited with code {self._process.returncode}; see {self.log_path}"
            )
        # A direct connection: urllib would send this through an HTTP proxy from the
        # environment. Short, since the window's thread waits for it.
        probe = http.client.HTTPConnection("127.0.0.1", self.port, timeout=0.2)
        try:
            probe.request("GET", "/v1/models")
            response = probe.getresponse()
            response.read()
            return 200 <= response.status < 300  # 503 while loading is "not yet"
        except (OSError, http.client.HTTPException):  # not listening yet, or half-way up
            return False
        finally:
            probe.close()

    def stop(self) -> None:
        """Stop a running child and drop its log.

        A child that died on its own keeps the log, since the failure message
        names it.
        """
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if self.log_path and os.path.exists(self.log_path):
            os.unlink(self.log_path)
