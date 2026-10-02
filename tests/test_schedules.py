"""The scheduler on its own, with a clock the test controls.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

import time

import pytest

from zmart_ai_agent.schedules import Scheduler
from zmart_ai_agent.settings import SCHEDULE_MIN_SECONDS, SCHEDULES_MAX


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


def test_a_repeating_schedule_comes_due_again_and_a_one_off_is_removed(clock):
    scheduler = Scheduler(clock)
    scheduler.add("watch", "look", every_seconds=60)
    scheduler.add("later", "switch the light off", in_seconds=120)
    assert scheduler.pop_due() is None  # nothing is due yet
    assert [s["name"] for s in scheduler.listing()] == [
        "watch",
        "later",
    ]  # the one due first, first
    clock.now += 61
    assert scheduler.pop_due()["instruction"] == "look" and scheduler.pop_due() is None
    clock.now += 60
    fired = [scheduler.pop_due(), scheduler.pop_due()]  # both due: one per tick, earliest first
    assert [f["name"] for f in fired] == ["later", "watch"]
    assert [s["name"] for s in scheduler.listing()] == ["watch"]  # the one-off is gone


def test_a_clock_time_is_today_or_tomorrow(clock):
    clock.now = time.mktime((2026, 9, 28, 14, 0, 0, 0, 0, -1))
    scheduler = Scheduler(clock)
    later = scheduler.add("run", "start the plan", at="15:00")
    earlier = scheduler.add("early", "look", at="9:30")
    assert later["at"] == "15:00" and later["due_in_s"] == 3600
    assert earlier["at"] == "09:30" and earlier["due_in_s"] == (24 - 14 + 9) * 3600 + 1800


def test_cancel_by_name_or_all(clock):
    scheduler = Scheduler(clock)
    scheduler.add("a", "look", every_seconds=60)
    scheduler.add("b", "look", every_seconds=60)
    assert scheduler.cancel("missing") == []
    assert scheduler.cancel("a") == ["a"] and [s["name"] for s in scheduler.listing()] == ["b"]
    assert scheduler.cancel("all") == ["b"] and scheduler.listing() == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one of"),
        ({"every_seconds": 60, "at": "15:00"}, "exactly one of"),
        ({"every_seconds": SCHEDULE_MIN_SECONDS - 1}, "at least"),
        ({"in_seconds": "soon"}, "must be a number"),
        ({"at": "25:00"}, "HH:MM"),
        ({"at": "three"}, "HH:MM"),
    ],
)
def test_a_bad_request_says_what_is_wrong(clock, kwargs, message):
    with pytest.raises(ValueError, match=message):
        Scheduler(clock).add("x", "look", **kwargs)
    with pytest.raises(ValueError, match="needs a name"):
        Scheduler(clock).add("", "look", every_seconds=60)
    with pytest.raises(ValueError, match="cannot be a schedule's name"):
        Scheduler(clock).add("all", "look", every_seconds=60)


def test_at_most_so_many_schedules(clock):
    scheduler = Scheduler(clock)
    for n in range(SCHEDULES_MAX):
        scheduler.add(f"s{n}", "look", every_seconds=60)
    with pytest.raises(ValueError, match="at most"):
        scheduler.add("one more", "look", every_seconds=60)
    scheduler.add("s0", "look again", every_seconds=30)  # replacing one is fine
    assert len(scheduler.listing()) == SCHEDULES_MAX
