"""Requests and waiting, the plan checklist, the one clock, and what changed.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-09
License: MIT
"""

import json
import time

import pytest
from pydantic_ai.messages import UserPromptPart
from test_agent import position, talk, tool_results

from zmart_ai_agent.instructions import WAIT_NOTE
from zmart_ai_agent.requests import Requests
from zmart_ai_agent.settings import CONTINUATIONS_MAX, SCHEDULE_MIN_SECONDS, WAIT_MAX_S


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def state_sent(script, index=-1):
    """The state block the model was sent in its latest request, as data."""
    for message in reversed(script.requests[index]):
        for part in message.parts:
            if isinstance(part, UserPromptPart) and "<microscope_state>" in str(part.content):
                content = part.content.split("<microscope_state>")[1]
                return json.loads(content[: -len("</microscope_state>")])
    raise AssertionError("no state block was sent")


# -- the requests on their own ---------------------------------------------------------------


def test_a_typed_message_opens_a_request_and_machine_turns_belong_to_it():
    clock = Clock()
    requests = Requests(clock)
    first = requests.typed("centre the sample, then focus")
    assert first.number == 1 and requests.open() is first
    requests.finish_turn("- [ ] centre\n- [x] focus\nDone so far.", 120)
    assert first.turns == 1 and first.tokens == 120 and first.plan == ["[ ] centre", "[x] focus"]
    clock.now += 90
    assert first.brief(clock.now)["minutes"] == 1.5 and first.brief(clock.now)["plan"]
    assert requests.machine(1) is first  # a schedule the request set fires: the same request
    stray = requests.machine(7)  # a schedule from before Clear context: a request of its own
    assert stray.number == 2 and stray.prompt == "(scheduled)"
    assert requests.typed("hello").number == 3


def test_a_wait_comes_due_on_the_clock_and_continues_the_request():
    clock = Clock()
    requests = Requests(clock)
    request = requests.typed("let it settle, then look")
    assert requests.due() is None
    pending = requests.wait(120)
    assert pending["seconds"] == 120 and requests.is_waiting() and requests.waiting is request
    assert "waiting" in request.brief(clock.now)
    clock.now += 119
    assert requests.due() is None
    clock.now += 1
    due, result = requests.due()
    assert due is request and result == "waited 120 s"
    assert not requests.is_waiting() and request.continuations == 1 and requests.due() is None


@pytest.mark.parametrize(
    ("seconds", "message"),
    [
        (SCHEDULE_MIN_SECONDS - 1, "wait at least"),
        (WAIT_MAX_S + 1, "wait at most"),
        ("soon", "must be a number"),
    ],
)
def test_a_bad_wait_says_what_is_wrong(seconds, message):
    requests = Requests(Clock())
    requests.typed("wait")
    with pytest.raises(ValueError, match=message):
        requests.wait(seconds)


def test_one_wait_at_a_time_and_a_limit_on_continuations():
    requests = Requests(Clock())
    with pytest.raises(ValueError, match="no request"):
        requests.wait(10)
    request = requests.typed("wait twice")
    requests.wait(10)
    with pytest.raises(ValueError, match="already waits"):
        requests.wait(10)
    request.continuations = CONTINUATIONS_MAX
    request.wait = None
    with pytest.raises(ValueError, match="tell the operator where it stands"):
        requests.wait(10)


def test_ending_a_request_drops_its_wait():
    requests = Requests(Clock())
    requests.typed("wait")
    requests.wait(10)
    requests.end("stopped")
    assert requests.open() is None and requests.waiting is None and requests.due() is None


# -- through the agent -------------------------------------------------------------------------


def test_wait_ends_the_turn_and_the_request_continues_with_its_plan(microscope):
    clock = Clock()
    microscope.scheduler.clock = clock
    steps = [
        ("wait", {"seconds": 60}),
        ("move_stage", {"x": 10}),  # the turn has asked to wait: refused
        "- [x] wait a minute\n- [ ] look\nWaiting a minute.",
        ("look", {"question": "settled?"}),
        "- [x] wait a minute\n- [x] look\nIt has settled.",
    ]
    conversation, script = talk(microscope, *steps)
    conversation.send("wait a minute, then look")
    waited, refused = tool_results(conversation)
    assert waited["waiting"] == {"seconds": 60} and waited["note"] == WAIT_NOTE
    assert refused["error"]["code"] == "waiting" and position(microscope)["x"] == 0.0
    request = microscope.requests.open()
    assert request.plan == ["[x] wait a minute", "[ ] look"] and request.wait is not None
    assert microscope.requests.due() is None
    clock.now += 60
    due, result = microscope.requests.due()
    conversation.send(
        f"[continuation of request {due.number}] {result}", "continuation", due.number
    )
    state = state_sent(script)
    assert state["request"]["number"] == 1 and state["request"]["turn"] == 1
    assert state["request"]["plan"] == ["[x] wait a minute", "[ ] look"]
    assert microscope.requests.open().plan == ["[x] wait a minute", "[x] look"]
    assert microscope.turn == 1  # a continuation is not the operator's turn
    assert len(microscope.frames.frames) == 1


def test_a_schedule_remembers_the_request_that_set_it(microscope):
    steps = [("schedule", {"name": "watch", "instruction": "look", "every_seconds": 60}), "Set."]
    conversation, _ = talk(microscope, *steps)
    conversation.send("look every minute")
    microscope.scheduler.clock = lambda: time.time() + 61
    assert microscope.scheduler.pop_due()["request"] == 1


def test_the_state_carries_the_request_only_when_it_says_something(microscope):
    conversation, script = talk(microscope, "Hello.", "Hi again.")
    conversation.send("hi")
    assert "request" not in state_sent(script)  # typed, no plan, no wait: the words follow anyway
    conversation.send("[scheduled 'x'] hi", "scheduled", 1)
    assert state_sent(script)["request"]["prompt"] == "hi"


def test_the_one_clock_is_the_schedulers(microscope):
    clock = Clock(time.mktime((2026, 10, 9, 14, 30, 0, 0, 0, -1)))
    microscope.scheduler.clock = clock
    conversation, script = talk(microscope, ("look", {"question": "?"}), "Looked.")
    conversation.send("look")
    assert state_sent(script)["clock"] == "14:30:00"
    assert tool_results(conversation)[0]["frame"]["time"] == "14:30:00"
    assert microscope.now() == clock.now


# -- every acting tool says what changed ------------------------------------------------------


def test_an_acting_tool_ends_with_what_changed_since_the_model_last_saw(microscope):
    steps = [
        ("set_microscope", {"settings": {"exposure_ms": 50}}),
        ("focus", {}),  # the routine moves z: the answer says so
        ("get_status", {}),  # a reading tool says nothing about changes
        "Done.",
    ]
    microscope.session.set_xyz(0, 0, 6)  # out of focus, so the routine has somewhere to go
    conversation, _ = talk(microscope, *steps)
    conversation.send("exposure 50, then focus")
    set_, focused, status = tool_results(conversation)
    assert set_["state_changed"] == {"settings.exposure_ms": 50.0}
    assert list(focused["state_changed"]) == ["position.z"]
    assert focused["state_changed"]["position.z"] == position(microscope)["z"] != 6.0
    assert "state_changed" not in status
    # a new message: what the model saw is the state block, so a move that changes
    # nothing says nothing
    conversation, _ = talk(microscope, ("move_stage", {"z": position(microscope)["z"]}), "Same.")
    conversation.send("stay")
    assert "state_changed" not in tool_results(conversation)[0]


def test_a_run_adds_its_images_to_the_frames(microscope):
    plan = {"name": "two", "channels": [{"name": "a"}, {"name": "b"}]}
    run = ("run_acquisition", {"plan_id": "two-1"})
    conversation, _ = talk(microscope, ("plan_acquisition", plan), "Start?", run, "Saved.")
    conversation.send("two images")
    conversation.send("yes")
    result = tool_results(conversation)[-1]
    assert result["frames"] == "1-2" and len(microscope.frames.frames) == 2
    assert microscope.frames.frames[0]["source"] == "run two-1: here_a"
