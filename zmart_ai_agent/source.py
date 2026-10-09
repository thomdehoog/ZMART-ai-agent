"""Reading the source: the search_source and read_source tools.

The agent can read its own code, the ZMART Controller's, and the connected
driver's, and nothing else on the computer. So when the operator asks how a
move reaches the microscope, the agent looks it up and shows the lines
rather than answering from memory.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-09
License: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import zmart_controller
from pydantic_ai import RunContext

from .instructions import OPTIONS_ADVICE
from .microscope import Microscope
from .settings import SOURCE_LINES, SOURCE_MATCHES
from .tooling import guarded_tool, refusal


@guarded_tool
def search_source(ctx: RunContext[Microscope], text: str) -> dict[str, Any]:
    """Search the source code of this agent, of the ZMART Controller and of
    this microscope's driver for a word or phrase, to explain how something works.

    Returns matching lines as "file:line: text". When nothing matches, returns
    the list of files that can be read instead.

    Args:
        text: the word or phrase to find, for example "def set_xyz" or
            "autofocus"; upper and lower case do not matter.
    """
    files = source_files(ctx.deps)
    matches = [
        f"{name}:{number}: {line.strip()[:160]}"
        for name, path in files.items()
        for number, line in enumerate(_lines(path), start=1)
        if text.lower() in line.lower()
    ]
    if not matches:
        return {"matches": [], "files": list(files)}
    return {"matches": matches[:SOURCE_MATCHES], "more": max(0, len(matches) - SOURCE_MATCHES)}


@guarded_tool
def read_source(
    ctx: RunContext[Microscope], file: str, start_line: int = 1, lines: int = 80
) -> dict[str, Any]:
    """Read part of a source file of this agent, the controller or the driver,
    with line numbers.

    Args:
        file: a file as search_source names it, for example
            "zmart_controller/zmart_controller.py".
        start_line: the first line to read, counting from 1.
        lines: how many lines to read, at most 200.
    """
    files = source_files(ctx.deps)
    if file not in files:
        message = f"{file!r} is not a source file here"
        return refusal(ctx, "not_found", message, OPTIONS_ADVICE, configured_options=list(files))
    text = _lines(files[file])
    start = max(1, start_line)
    chunk = text[start - 1 : start - 1 + max(1, min(lines, SOURCE_LINES))]
    return {
        "file": file,
        "lines": f"{start} to {start + len(chunk) - 1} of {len(text)}",
        "text": "\n".join(f"{start + i}: {line}" for i, line in enumerate(chunk)),
    }


def source_roots(microscope: Microscope) -> dict[str, Path]:
    """The source the agent may read: itself, the controller, and the connected
    microscope's driver when the controller knows where it lives. Nothing else."""
    roots = {
        "zmart_ai_agent": Path(__file__).resolve().parent,
        "zmart_controller": Path(zmart_controller.__file__).resolve().parent,
    }
    driver = microscope.driver_folder()
    # A driver inside the controller, such as its mock, is already readable there.
    if driver is not None and not any(driver.is_relative_to(r) for r in roots.values()):
        roots[driver.name] = driver
    return roots


def source_files(microscope: Microscope) -> dict[str, Path]:
    """The files the agent may read, by name: "zmart_controller/zmart_controller.py", ..."""
    return {
        f"{label}/{path.relative_to(root).as_posix()}": path
        for label, root in source_roots(microscope).items()
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()
