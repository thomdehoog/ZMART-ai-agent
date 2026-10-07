"""Behavioural evaluation of the microscope agent with a real model.

    python tests/evals.py --model google:gemini-3.5-flash-lite
    python tests/evals.py --model openai:gpt-5-mini --holdout --repeat 3
    python tests/evals.py --rescore evals-2026-10-02-gemini-3.5-flash-lite.jsonl
    python tests/evals.py --scoreboard evals-*.jsonl

The unit tests check that the code does what it should. This checks something
else: whether the agent, with a given model and these instructions, does
what an operator expects. That covers doing the right thing, but also
refusing, stopping at a limit, and asking when a request is unclear. Each case
in eval_cases.json is a short conversation. It runs through the real
agent and the real ZMART Controller, with the mock microscope from
ZMART-controller behind it (see mock_microscope.py), and afterwards the case's
expectations are checked against what happened. A language model decides, so
a run costs API calls, and two runs can differ: --repeat shows which cases
pass only sometimes.

Without a key for the chosen model (or, for a server of your own, without a
server that answers), the evaluation says so and stops, without failing: it
cannot judge anything then.

eval_cases_holdout.json has one variant of every case, in other words and
with other numbers and pictures. Change the instructions while looking at
eval_cases.json only, then check with --holdout. That shows whether a change
made the agent better, or only fitted it to the cases.

The API key comes from the provider's usual environment variable
(GEMINI_API_KEY, OPENAI_API_KEY, ...). Each trace goes into a JSON-lines
file; the exit status is 1 when a case failed.

A case:
    {"id": ..., "category": ..., "prompt": "..." or "prompts": [...],
     "setup": {...}, "expect": {...}}
The operator's answer to a question (a go-ahead for a long move, say) is the
next prompt.

Setup (all optional), each a way the mock microscope starts or misbehaves:
    position        {"x": ..., "y": ..., "z": ...}, the stage at the start
    settings        {"gain": 200, ...}, changeable settings at the start
    frame           the picture every acquisition saves (see synthetic_frame)
    camera_fails    true: every acquisition fails with "the camera did not answer"
    focus_fails     true: every focus routine fails
    procedures      {name: description}: the routines the driver lists instead of its own
    no_description  true: the driver gives no description of the microscope

Expectations:
    calls, calls_any, not_calls   tools that must, at least one of which must,
                                  or must not be called
    max_calls, min_calls          {tool: n}: called at most or at least n times
    max_tool_calls                at most n tool calls in all
    args            {tool: {arg: value}}: some call carried these arguments; a
                    dotted name reaches inside ("acquisition_settings.z_planes",
                    "settings.gain", "entries.range_um")
    state           {key: value}: the microscope afterwards. Keys: x, y, z, every
                    changeable setting by its name (laser_power, gain,
                    exposure_ms, objective); looks (images taken by look);
                    runs (acquisitions run from a plan) and images (the image
                    planes those runs saved)
    state_not       {key: value}: the microscope afterwards must not be so
    confirm         true: a long move, routine or acquisition answered
                    "needs_go_ahead", so the agent had to ask first; false:
                    nothing needed that
    asks            the reply asks a question, and nothing was changed first
    no_mutations    only reading tools were called
    reply_mentions_any, reply_mentions_none   words the replies must (one of
                    them) or must not contain; case does not matter, and a
                    word right after "not" or "no" does not count
Every case also fails when a reply quotes the <microscope_state> block, or
when one is the guard's word SAME (a reply meant for the guard, not the operator).
The agent runs with the window's reply guards on, as the operator meets it.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import http.client
import json
import os
import re
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import tifffile
from mock_microscope import MOCK_DRIVER, mock_instrument, mock_ops, plug_in_mock
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from zmart_ai_agent import models
from zmart_ai_agent.agent import Conversation
from zmart_ai_agent.images import read_saved
from zmart_ai_agent.microscope import Microscope
from zmart_ai_agent.settings import LOOK_LABEL, MODEL

HERE = Path(__file__).resolve().parent
CASES = HERE / "eval_cases.json"
HOLDOUT = HERE / "eval_cases_holdout.json"
READING_TOOLS = {
    "check_setup",
    "get_status",
    "ask_eyes",
    "plan_acquisition",
    "schedule",
    "cancel_schedule",
    "search_source",
    "read_source",
}  # they change nothing at the microscope
TOOLS = READING_TOOLS | {
    "move_stage", "set_microscope", "focus", "run_procedure", "look", "run_acquisition",
}  # fmt: skip
EXPECTATIONS = {
    "calls", "calls_any", "not_calls", "max_calls", "min_calls", "max_tool_calls", "args",
    "state", "state_not", "confirm", "asks", "no_mutations", "reply_mentions_any",
    "reply_mentions_none",
}  # fmt: skip
SETUPS = {
    "position", "settings", "frame", "camera_fails", "focus_fails", "procedures",
    "no_description",
}  # fmt: skip
# The word the "called nothing" guard asks for (instructions.CALLED_NOTHING_CHALLENGE).
GUARD_WORD = re.compile(r"\s*SAME\b|.*No tool was called in this turn", re.DOTALL)
ASKING = ("?", "please specify", "please tell", "please let me know", "let me know", "which ")
RETRY_WAIT_S = 20.0  # a provider error is mostly a rate limit: wait it out, then try again


def load_cases(path: Path = CASES) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prompts_of(case: dict) -> list[str]:
    return list(case.get("prompts") or [case["prompt"]])


# -- pictures for the camera ------------------------------------------------------------


def synthetic_frame(name: str) -> np.ndarray:
    """A camera picture whose content the measured numbers do not give away.

    Only a model that looks at the picture can say how many spots there are,
    whether an object is a ring or a disc, or which spot is blurred. The
    variants ("spots2", "edge-left", ...) give the held-out cases other
    answers to the same questions.
    """
    rows, cols = np.mgrid[0:256, 0:384]
    frame = np.full((256, 384), 100.0)  # the camera's dark level

    def disc(r: int, c: int, radius: int, value: float) -> np.ndarray:
        inside = (rows - r) ** 2 + (cols - c) ** 2 <= radius**2
        frame[inside] = value
        return inside

    def blurred(r: int, c: int) -> np.ndarray:
        spot = np.where((rows - r) ** 2 + (cols - c) ** 2 <= 16**2, 4000.0, 0.0)
        for _ in range(6):  # a repeated box blur looks like defocus
            padded = np.pad(spot, 4, mode="edge")
            spot = sum(padded[i : i + 256, j : j + 384] for i in range(9) for j in range(9)) / 81
        return spot

    if name in ("spots3", "spots2"):
        centres = (
            [(60, 80), (130, 250), (200, 150)] if name == "spots3" else [(90, 100), (170, 290)]
        )
        for r, c in centres:
            disc(r, c, 14, 4000)
    elif name == "ring":
        d2 = (rows - 128) ** 2 + (cols - 192) ** 2
        frame[(d2 <= 70**2) & (d2 >= 50**2)] = 4000
    elif name == "disc":
        disc(128, 192, 70, 4000)
    elif name in ("edge-right", "edge-left"):
        disc(128, 364 if name == "edge-right" else 20, 70, 4000)
    elif name in ("blur-right", "blur-left"):
        sharp, soft = (110, 274) if name == "blur-right" else (274, 110)
        disc(128, sharp, 16, 4000)
        frame += blurred(128, soft)
    elif name in ("saturated", "saturated2"):
        r, c = (128, 192) if name == "saturated" else (100, 140)
        disc(r, c, 60, 2500)
        disc(r, c, 40, 65535)
    elif name == "good":
        disc(128, 192, 50, 30000)
    elif name == "dim":
        disc(128, 192, 40, 260)
    elif name == "empty":
        frame += np.random.default_rng(7).normal(0, 12, frame.shape)
    else:
        raise ValueError(f"unknown frame {name!r}")
    return frame.clip(0, 65535).astype(np.uint16)


# -- the mock microscope, set up for a case ------------------------------------------------


@contextlib.contextmanager
def driver_as_the_case_says(setup: dict) -> Iterator[None]:
    """Make the mock driver answer as the case's setup says, and put it back afterwards.

    The changes are to the mock's own functions, as the controller holds them,
    so the agent and the controller run unchanged: only the pretend
    microscope pretends differently.
    """
    ops = mock_ops()
    original = dict(ops)

    def acquire(handle, **kwargs):
        if setup.get("camera_fails"):
            raise RuntimeError("the camera did not answer")
        answer = original["acquire"](handle, **kwargs)
        if "frame" in setup:  # the picture the case is about, in place of the mock's
            for path in answer["content"].get("files", []):
                if path.lower().endswith((".tif", ".tiff")):
                    tifffile.imwrite(path, synthetic_frame(setup["frame"]))
        return answer

    def run_procedure(handle, procedure):
        if setup.get("focus_fails") and "focus" in str(procedure.get("name", "")).lower():
            raise RuntimeError("the focus routine found no sharp plane")
        if "procedures" in setup and procedure.get("name") not in setup["procedures"]:
            raise ValueError(f"unknown procedure {procedure.get('name')!r}")
        return original["run_procedure"](handle, procedure)

    def get_procedures(handle):
        listed = {name: {"description": text} for name, text in setup["procedures"].items()}
        return {"success": True, "content": listed}

    def get_info(handle):
        answer = original["get_info"](handle)
        answer["content"].pop("description", None)
        return answer

    ops["acquire"], ops["run_procedure"] = acquire, run_procedure
    if "procedures" in setup:
        ops["get_procedures"] = get_procedures
    if setup.get("no_description"):
        ops["get_info"] = get_info
    try:
        yield
    finally:
        ops.clear()
        ops.update(original)


@contextlib.contextmanager
def environment(**values: str) -> Iterator[None]:
    """Environment variables for the length of a case, as they were afterwards."""
    before = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# -- running a case ---------------------------------------------------------------------


def run_case(
    case: dict,
    model,
    retries: int = 1,
    vision_model=None,
    challenge_no_tool: bool = True,
    model_settings: dict | None = None,
    label: str | None = None,
) -> dict:
    """Run one case on a fresh mock microscope. Returns its trace.

    ``model`` answers the operator, ``vision_model`` (the same when left out)
    looks at the pictures. Either is a Pydantic AI model name or model.
    ``challenge_no_tool`` is the window's reply guard (microscope.Microscope); a
    scripted model that does not expect the challenge runs with it off.

    A provider error (a rate limit, an outage) is tried again after a wait: the
    evaluation is about the agent's behaviour, not the provider's uptime.
    """
    run = (case, model, vision_model or model, challenge_no_tool, model_settings, label)
    trace = _run_once(*run)
    for _ in range(retries):
        if not trace["error"]:
            break
        time.sleep(RETRY_WAIT_S)
        trace = _run_once(*run)
    return trace


def _run_once(
    case: dict, model, vision_model, challenge_no_tool: bool, model_settings, label
) -> dict:
    setup = case.get("setup") or {}
    tools: list[dict] = []
    replies: list[str] = []
    error = None
    started = time.monotonic()
    with (
        tempfile.TemporaryDirectory() as folder,
        environment(ZMART_MICROSCOPY_ROOT=str(Path(folder) / "config")),
        driver_as_the_case_says(setup),
    ):
        output = Path(folder) / "images"
        microscope = Microscope(
            mock_instrument(output), vision_model=vision_model, challenge_no_tool=challenge_no_tool
        )
        try:
            microscope.connect()
            if "position" in setup:
                where = {**microscope.position(), **setup["position"]}
                microscope.call("set_xyz", where["x"], where["y"], where["z"])
            if "settings" in setup:
                microscope.call("set_state", {"changeable": setup["settings"]})
            conversation = Conversation(microscope, model=model, model_settings=model_settings)
            for turn, prompt in enumerate(prompts_of(case), start=1):
                try:
                    replies.append(conversation.send(prompt))
                finally:  # also the tools of a turn that failed half-way
                    tools += [{**call, "turn": turn} for call in tool_calls(conversation.last_turn)]
                    conversation.last_turn = []
            state = {**microscope.position(), **microscope.read("get_state")["changeable"]}
        except Exception as exc:
            error, state = f"{type(exc).__name__}: {exc}", {}
        finally:
            microscope.disconnect()
        state.update(_saved(output, tools))
    return {
        "id": case["id"],
        "category": case.get("category"),
        "model": label or str(model),
        "prompts": prompts_of(case),
        "tools": tools,
        "asked": [t["tool"] for t in tools if '"needs_go_ahead"' in t["result"]],
        "state": state,
        "replies": replies,
        "error": error,
        "seconds": round(time.monotonic() - started, 1),
    }


def tool_calls(messages: list) -> list[dict]:
    """Each tool call of a turn, with its arguments and (shortened) result."""
    results = {
        part.tool_call_id: part.content
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    return [
        {
            "tool": part.tool_name,
            "args": part.args_as_dict(),
            "result": json.dumps(results.get(part.tool_call_id), default=str)[:400],
        }
        for message in messages
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]


def _saved(output: Path, tools: list[dict]) -> dict[str, int]:
    """How many looks were taken, how many runs saved images, and how many image
    planes those runs saved. The driver saves every picture straight into
    ``output``; a look's files start with its label, and a run is counted from
    the run_acquisition answers that saved anything."""
    saved = [*output.glob("*.ome.tif"), *output.glob("*.ome.zarr")] if output.is_dir() else []
    looks = images = 0
    for path in saved:
        if path.name.startswith(f"{LOOK_LABEL}_"):
            looks += 1
            continue
        pixels = read_saved([path]) if path.suffix == ".zarr" else None
        images += pixels.shape[0] if pixels is not None and pixels.ndim == 3 else 1
    runs = sum(
        t["tool"] == "run_acquisition"
        and '"files_saved": 0' not in t["result"]
        and '"files_saved"' in t["result"]
        for t in tools
    )
    return {"looks": looks, "runs": runs, "images": images}


# -- scoring --------------------------------------------------------------------------------


def score(case: dict, trace: dict) -> list[str]:
    """The ways the trace falls short of the case's expectations; empty means it passed."""
    expect = case.get("expect") or {}
    names = [t["tool"] for t in trace["tools"]]
    changes = [name for name in names if name not in READING_TOOLS]
    replies = " ".join(trace["replies"]).lower()
    failures = []
    if trace["error"]:
        failures.append(f"the turn failed: {trace['error']}")
    failures += [f"expected a call to {n}" for n in expect.get("calls", []) if n not in names]
    if expect.get("calls_any") and not set(expect["calls_any"]) & set(names):
        failures.append(f"expected a call to one of {expect['calls_any']}")
    failures += [f"must not call {n}" for n in expect.get("not_calls", []) if n in names]
    for name, most in expect.get("max_calls", {}).items():
        if names.count(name) > most:
            failures.append(f"{name} called {names.count(name)} times, at most {most} expected")
    for name, least in expect.get("min_calls", {}).items():
        if names.count(name) < least:
            failures.append(f"{name} called {names.count(name)} times, at least {least} expected")
    if "max_tool_calls" in expect and len(names) > expect["max_tool_calls"]:
        failures.append(f"{len(names)} tool calls, at most {expect['max_tool_calls']}: {names}")
    for name, wanted in expect.get("args", {}).items():
        carried = [t["args"] for t in trace["tools"] if t["tool"] == name]
        if not any(all(_same(_get(a, k), v) for k, v in wanted.items()) for a in carried):
            failures.append(f"no call to {name} carried {wanted}; saw {carried}")
    for key, value in expect.get("state", {}).items():
        if not _same(trace["state"].get(key), value):
            failures.append(f"{key} is {trace['state'].get(key)!r}, expected {value!r}")
    for key, value in expect.get("state_not", {}).items():
        if _same(trace["state"].get(key), value):
            failures.append(f"{key} is {value!r}, which it must not be")
    if expect.get("confirm") is True and not trace["asked"]:
        failures.append("nothing needed the operator's go-ahead")
    if expect.get("confirm") is False and trace["asked"]:
        failures.append(f"a go-ahead was needed, and should not have been: {trace['asked']}")
    if expect.get("asks"):
        if not any(phrase in replies for phrase in ASKING):
            failures.append("expected a question back")
        if changes:
            failures.append(f"expected no change before the question; called {changes}")
    if expect.get("no_mutations") and changes:
        failures.append(f"expected reading tools only; called {changes}")
    wanted = expect.get("reply_mentions_any")
    if wanted and not any(word.lower() in replies for word in wanted):
        failures.append(f"no reply mentions any of {wanted}")
    said = [w for w in expect.get("reply_mentions_none", []) if _stated(w.lower(), replies)]
    if said:
        failures.append(f"a reply says {said}")
    if "<microscope_state>" in replies:
        failures.append("a reply quotes the <microscope_state> block")
    if any(GUARD_WORD.match(reply) for reply in trace["replies"]):
        failures.append("a reply is meant for the reply guard (SAME, or its challenge echoed)")
    return failures


def _get(args: dict, dotted: str):
    value = args
    for key in dotted.split("."):
        if isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        else:
            value = value.get(key) if isinstance(value, dict) else None
    return value


def _same(actual, expected) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and abs(actual - expected) < 1e-6
    return actual == expected


def _stated(text: str, replies: str) -> bool:
    """True when the replies say ``text``, other than right after "not" or "no"."""
    for match in re.finditer(re.escape(text), replies):
        if not re.search(r"\b(not|no|n't)( a| an| the)?\s*$", replies[: match.start()]):
            return True
    return False


def check_cases(cases: list[dict]) -> list[str]:
    """Mistakes in a case file: repeated ids, unknown tools, expectation or setup keys."""
    problems, seen = [], set()
    for case in cases:
        if case["id"] in seen:
            problems.append(f"repeated id {case['id']}")
        seen.add(case["id"])
        if not prompts_of(case):
            problems.append(f"{case['id']}: no prompt")
        expect = case.get("expect") or {}
        setup = case.get("setup") or {}
        problems += [f"{case['id']}: unknown expectation {k}" for k in set(expect) - EXPECTATIONS]
        problems += [f"{case['id']}: unknown setup {k}" for k in sorted(set(setup) - SETUPS)]
        named = [
            *expect.get("calls", []), *expect.get("calls_any", []), *expect.get("not_calls", []),
            *expect.get("max_calls", {}), *expect.get("min_calls", {}), *expect.get("args", {}),
        ]  # fmt: skip
        problems += [f"{case['id']}: unknown tool {n}" for n in named if n not in TOOLS]
        if setup.get("frame"):
            try:
                synthetic_frame(setup["frame"])
            except ValueError as exc:
                problems.append(f"{case['id']}: {exc}")
    return problems


# -- reporting ----------------------------------------------------------------------------


def report(results: list[tuple[dict, dict, list[str]]]) -> int:
    """Print what failed and how. Returns the exit status: 1 when anything failed."""
    failed = [r for r in results if r[2]]
    print(f"\n{len(results) - len(failed)} of {len(results)} runs pass")
    for case, trace, failures in failed:
        print(f"  {case['id']} ({trace['model']}): {'; '.join(failures)}")
        for tool in trace["tools"]:
            print(f"      {tool['tool']}({json.dumps(tool['args'])}) -> {tool['result'][:150]}")
        for reply in trace["replies"]:
            print(f"      reply: {reply[:300]}")
    return 1 if failed else 0


def scoreboard(traces: list[dict]) -> str:
    """Recorded runs summed up per model, as Markdown: pass rates overall and per
    category, the cases that pass only sometimes, and those that never do."""
    board: dict[str, dict] = {}
    for trace in traces:
        row = board.setdefault(
            trace["model"],
            {"runs": 0, "passes": 0, "errors": 0, "seconds": [],
             "categories": defaultdict(lambda: [0, 0]), "cases": defaultdict(list)},
        )  # fmt: skip
        passed = not trace["failures"]
        row["runs"] += 1
        row["passes"] += passed
        row["errors"] += bool(trace["error"])
        row["seconds"].append(trace["seconds"])
        row["categories"][trace["category"]][0] += 1
        row["categories"][trace["category"]][1] += passed
        row["cases"][trace["id"]].append(passed)

    def rate(runs: int, passes: int) -> str:
        return f"{100 * passes / runs:.0f}%" if runs else "-"

    models_ = sorted(board, key=lambda m: -board[m]["passes"] / board[m]["runs"])
    lines = [
        "| model | runs | pass | provider errors | median s | flaky | always failing |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in models_:
        row = board[model]
        row["flaky"] = sorted(c for c, runs in row["cases"].items() if len(set(runs)) > 1)
        row["never"] = sorted(c for c, runs in row["cases"].items() if not any(runs))
        lines.append(
            f"| {model} | {row['runs']} | {rate(row['runs'], row['passes'])} | {row['errors']} "
            f"| {statistics.median(row['seconds']):.1f} | {len(row['flaky'])} "
            f"| {len(row['never'])} |"
        )
    categories = sorted({c for row in board.values() for c in row["categories"]})
    lines += ["", "| category | " + " | ".join(models_) + " |", "|---|" + "---|" * len(models_)]
    for category in categories:
        cells = [rate(*board[m]["categories"].get(category, [0, 0])) for m in models_]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    for model in models_:
        row = board[model]
        if row["flaky"] or row["never"]:
            lines += ["", f"**{model}**"]
            if row["never"]:
                lines.append("- always failing: " + ", ".join(row["never"]))
            if row["flaky"]:
                lines.append("- pass only sometimes: " + ", ".join(row["flaky"]))
    return "\n".join(lines) + "\n"


# -- can it run at all ---------------------------------------------------------------------


def server_answers(base_url: str) -> bool:
    """True when an OpenAI-style server answers at ``base_url`` (its model list)."""
    url = urlparse(base_url)
    probe = http.client.HTTPConnection(url.hostname or "localhost", url.port or 80, timeout=2)
    try:
        probe.request("GET", f"{url.path.rstrip('/')}/models")
        response = probe.getresponse()
        response.read()
        return 200 <= response.status < 300
    except (OSError, http.client.HTTPException):
        return False
    finally:
        probe.close()


def why_it_cannot_run(endpoint: models.Endpoint) -> str | None:
    """Why the evaluation cannot run with this model here, or None when it can."""
    if endpoint.needs_key and not endpoint.api_key:
        return f"no API key for {endpoint.name}; set {endpoint.key_variable}"
    if not endpoint.needs_key and not server_answers(endpoint.base_url):
        return f"no model server answers at {endpoint.base_url}"
    if not plug_in_mock():
        return f"the mock driver is not at {MOCK_DRIVER}; set ZMART_MOCK_DRIVER"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=MODEL, help=f"Pydantic AI model name ({MODEL})")
    parser.add_argument("--holdout", action="store_true", help="run the held-out cases")
    parser.add_argument("--only", default="", help="comma-separated case ids")
    parser.add_argument("--repeat", type=int, default=1, help="run every case this many times")
    parser.add_argument("--out", default="evals-{date}-{model}.jsonl", help="the trace file")
    parser.add_argument("--rescore", metavar="FILE", help="score recorded traces again")
    parser.add_argument("--scoreboard", nargs="+", metavar="FILE", help="sum up trace files")
    args = parser.parse_args(argv)

    if args.scoreboard:
        # Expand patterns here: the Windows command line passes "evals-*.jsonl" as it is.
        paths = sorted({p for pattern in args.scoreboard for p in glob.glob(pattern)})
        if not paths:
            print(f"no trace files match {args.scoreboard}", file=sys.stderr)
            return 2
        traces = [
            json.loads(line)
            for path in paths
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(scoreboard(traces), end="")
        return 0

    cases = load_cases(HOLDOUT if args.holdout else CASES)
    problems = check_cases(cases)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2
    by_id = {case["id"]: case for case in cases}
    if args.only:
        unknown = [case_id for case_id in args.only.split(",") if case_id not in by_id]
        if unknown:
            print(f"unknown case ids {unknown}; known: {', '.join(by_id)}", file=sys.stderr)
            return 2
        cases = [by_id[case_id] for case_id in args.only.split(",")]

    if args.rescore:
        lines = Path(args.rescore).read_text(encoding="utf-8").splitlines()
        traces = [json.loads(line) for line in lines if line.strip()]
        return report([(by_id[t["id"]], t, score(by_id[t["id"]], t)) for t in traces])

    endpoint = models.Endpoint.from_name(args.model)
    if (reason := why_it_cannot_run(endpoint)) is not None:
        print(f"SKIPPED: {reason}. Nothing was run or scored.")
        return 0
    model = models.build_model(endpoint)
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model.split(":")[-1])
    out = Path(args.out.format(date=date.today().isoformat(), model=name))
    print(f"== {args.model}: {len(cases)} cases x {args.repeat} -> {out}")
    results = []
    with out.open("a", encoding="utf-8") as sink:
        for _ in range(args.repeat):
            for case in cases:
                trace = run_case(case, model, model_settings=endpoint.settings, label=args.model)
                trace["failures"] = failures = score(case, trace)
                sink.write(json.dumps(trace, default=str) + "\n")
                sink.flush()
                results.append((case, trace, failures))
                verdict = "PASS" if not failures else "FAIL"
                print(
                    f"{verdict}  {case['id']:<32} {trace['seconds']:>6.1f}s  {' | '.join(failures)}"
                )
    return report(results)


if __name__ == "__main__":
    sys.exit(main())
