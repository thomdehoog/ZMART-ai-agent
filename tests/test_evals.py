"""The evaluation harness itself, offline: the case files are sound, a scripted
model that does what a case expects passes, one that does not fails with
reasons, and without a model key the evaluation skips cleanly. No real model
is called here.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

import socket

import evals
from test_agent import Script

from zmart_ai_agent.images import image_statistics


def case(case_id, path=evals.CASES):
    return next(c for c in evals.load_cases(path) if c["id"] == case_id)


def run(case_id, *steps, vision="A plain grey field."):
    """Run a case with scripted models: one answers, the other looks at the picture."""
    model, eyes = Script(*steps).model(), Script(vision).model()
    return evals.run_case(case(case_id), model, 0, eyes, challenge_no_tool=False)


def test_the_case_files_are_sound_and_the_holdout_mirrors_the_cases():
    cases, held = evals.load_cases(evals.CASES), evals.load_cases(evals.HOLDOUT)
    assert evals.check_cases(cases) == [] and evals.check_cases(held) == []
    assert [c["category"] for c in cases] == [c["category"] for c in held]  # case for case
    prompts = {p for c in cases for p in evals.prompts_of(c)}
    assert not prompts & {p for c in held for p in evals.prompts_of(c)}  # other words


def test_a_case_with_a_mistake_is_caught():
    bad = [
        {"id": "a", "category": "x", "prompt": "hi", "expect": {"calls": ["teleport"]}},
        {"id": "a", "category": "x", "prompt": "hi", "expect": {"wishes": True}},
        {"id": "b", "category": "x", "prompt": "hi", "setup": {"frame": "nope", "magic": 1}},
    ]
    problems = evals.check_cases(bad)
    assert "a: unknown tool teleport" in problems and "repeated id a" in problems
    assert "a: unknown expectation wishes" in problems and "b: unknown setup magic" in problems
    assert any("unknown frame 'nope'" in p for p in problems)


def test_the_frames_say_what_their_cases_assume():
    saturated = image_statistics(evals.synthetic_frame("saturated"))["saturated_percent"]
    assert saturated > 1
    assert image_statistics(evals.synthetic_frame("good"))["saturated_percent"] == 0
    assert image_statistics(evals.synthetic_frame("dim"))["max"] < 1000


def test_a_model_that_does_the_right_thing_passes():
    trace = run("move-x-um", ("move_stage", {"x": 200}), "The stage is at x = 200 um.")
    assert evals.score(case("move-x-um"), trace) == []
    assert trace["tools"][0]["tool"] == "move_stage" and trace["state"]["x"] == 200


def test_a_model_that_does_not_fails_with_reasons():
    trace = run("move-x-um", "Sure, done.")
    failures = evals.score(case("move-x-um"), trace)
    assert "expected a call to move_stage" in failures and "x is 0.0, expected 200" in failures


def test_a_long_move_is_asked_about_and_the_answer_is_the_next_prompt():
    long = ("move_stage", {"x": 3000})
    trace = run("move-long-cancelled", long, "Shall I move 3 mm to x = 3 mm?", "OK, we stay.")
    assert trace["asked"] == ["move_stage"] and trace["state"]["x"] == 0
    assert evals.score(case("move-long-cancelled"), trace) == []


def test_asking_back_passes_only_without_a_change():
    asked = run("ambiguous-move", "Which axis, and how far?")
    assert evals.score(case("ambiguous-move"), asked) == []
    moved = run("ambiguous-move", ("move_stage", {"x": 10}), "Moved a bit, is that ok?")
    assert any("no change" in f for f in evals.score(case("ambiguous-move"), moved))


def test_a_setting_case_reads_the_microscope_back():
    call = ("set_microscope", {"settings": {"exposure_ms": 50}})
    trace = run("set-exposure-ms", call, "The exposure is 50 ms.")
    assert trace["state"]["exposure_ms"] == 50 and trace["state"]["gain"] == 100
    assert evals.score(case("set-exposure-ms"), trace) == []


def test_a_vision_case_shows_the_synthetic_frame():
    trace = run(
        "vision-count",
        ("look", {"question": "how many spots?"}),
        "I see three bright spots.",
        vision="Three separate bright spots.",
    )
    assert evals.score(case("vision-count"), trace) == []
    assert '"max": 4000.0' in trace["tools"][0]["result"]  # the picture, not the mock's own


def test_a_camera_that_fails_and_a_driver_without_focus_or_description():
    trace = run("camera-failure-proposes", ("look", {"question": "what?"}), "The camera failed.")
    assert "the camera did not answer" in trace["tools"][0]["result"]
    trace = run("focus-none", ("focus", {}), "This microscope cannot focus by itself.")
    assert "lists no focus procedure" in trace["tools"][0]["result"]
    assert evals.score(case("focus-none"), trace) == []
    trace = run("no-description", "The driver does not describe the settings.")
    assert evals.score(case("no-description"), trace) == []


def test_an_acquisition_case_counts_the_saved_images():
    plan = {
        "plan": {
            "name": "two",
            "channels": [
                {"name": "dim", "settings": {"laser_power": 5}},
                {"name": "bright", "settings": {"laser_power": 20}},
            ],
        }
    }
    trace = run(
        "acquisition-two-channels",
        ("plan_acquisition", plan),
        "Two images, at laser power 5 and 20. Shall I start?",
        ("run_acquisition", {"plan_id": "two-1"}),
        "Two images saved.",
    )
    assert trace["state"]["runs"] == 1 and trace["state"]["images"] == 2
    assert evals.score(case("acquisition-two-channels"), trace) == []


def test_argument_names_reach_inside_the_plan():
    plan = {"channels": [{"name": "gfp"}], "acquisition_settings": {"z_planes": 5}}
    assert evals._get(plan, "channels.0.name") == "gfp"
    assert evals._get(plan, "acquisition_settings.z_planes") == 5
    assert evals._get(plan, "channels.5.name") is None


def test_negated_words_do_not_count_against_a_reply():
    assert not evals._stated("fully inside", "the sample is not fully inside the image")
    assert evals._stated("fully inside", "the sample is fully inside the image")


def test_the_scoreboard_sums_up_runs_per_model():
    traces = [
        {"id": "a", "category": "moves", "model": "m", "failures": [], "error": None, "seconds": 1},
        {
            "id": "a",
            "category": "moves",
            "model": "m",
            "failures": ["x"],
            "error": None,
            "seconds": 3,
        },
    ]
    board = evals.scoreboard(traces)
    assert "| m | 2 | 50% | 0 | 2.0 | 1 | 0 |" in board and "pass only sometimes: a" in board


def test_without_a_key_the_evaluation_skips_cleanly(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    for variable in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    assert evals.main(["--model", "openai:gpt-5-mini"]) == 0
    said = capsys.readouterr().out
    assert said.startswith("SKIPPED") and "OPENAI_API_KEY" in said
    assert evals.main(["--model", "google:gemini-3.5-flash-lite"]) == 0
    assert "GEMINI_API_KEY" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []  # no trace file for a run that did not happen


def test_without_a_model_server_the_evaluation_skips_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(evals, "server_answers", lambda url: False)
    assert evals.main(["--model", "gemma4:31b"]) == 0
    said = capsys.readouterr().out
    assert said.startswith("SKIPPED") and "localhost:11434" in said


def test_a_closed_port_is_no_server():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # free, and nobody listens once the probe closes
    assert evals.server_answers(f"http://127.0.0.1:{port}/v1") is False
