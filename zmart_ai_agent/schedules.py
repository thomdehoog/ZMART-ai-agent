"""Schedules: an instruction carried out later, as if the operator typed it then.

"Look every three minutes", "in ten minutes start the plan", "at 15:00 switch
the light off": the agent's ``schedule`` tool adds a named schedule here, and
the chat window's clock asks ``pop_due`` every second and sends each due
instruction as an ordinary turn, marked as scheduled in the transcript. So a
scheduled action goes through the same tools, checks and refusals as anything
typed, one at a time and never while a turn is running. The model cannot keep
time; this does, and the microscope state carries the clock and the schedules
so the model knows both.

Two threads use it at once: the agent's tools add and cancel schedules,
and the window's clock pops the due one. A lock makes every change whole, so
neither thread ever sees a half-changed list.

A scheduled turn is not the operator's: a scheduled acquisition or long stage
move still asks for their go-ahead in the chat, and waits until they answer.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from typing import Any

from .settings import CLOCK_FORMAT, SCHEDULE_MIN_SECONDS, SCHEDULES_MAX


class Scheduler:
    """Named schedules: repeating every so many seconds, once after a delay, or
    once at a clock time. ``clock`` is replaceable for the tests."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = {}

    def add(
        self,
        name: str,
        instruction: str,
        every_seconds: float | None = None,
        in_seconds: float | None = None,
        at: str | None = None,
    ) -> dict[str, Any]:
        """Add or replace the schedule ``name``; returns it as the listing shows it.

        Exactly one of ``every_seconds``, ``in_seconds`` and ``at`` must be given.
        A ValueError says what is wrong with the request.
        """
        name, instruction = (name or "").strip(), (instruction or "").strip()
        if not name or not instruction:
            raise ValueError("a schedule needs a name and an instruction")
        if name.lower() == "all":
            raise ValueError("'all' cannot be a schedule's name; cancel_schedule uses it")
        given = [
            (key, value)
            for key, value in (
                ("every_seconds", every_seconds),
                ("in_seconds", in_seconds),
                ("at", at),
            )
            if value is not None
        ]
        if len(given) != 1:
            raise ValueError("give exactly one of every_seconds, in_seconds or at")
        key, value = given[0]
        now = self.clock()
        item: dict[str, Any] = {"name": name, "instruction": instruction}
        if key == "at":
            item["at"] = _clock_time(value)
            item["next"] = _next_occurrence(item["at"], now)
        else:
            seconds = _seconds(value)
            item[key] = seconds
            item["next"] = now + seconds
        with self._lock:
            if name not in self._items and len(self._items) >= SCHEDULES_MAX:
                raise ValueError(f"at most {SCHEDULES_MAX} schedules; cancel one first")
            self._items[name] = item
        return self._listed(item, now)

    def cancel(self, name: str) -> list[str]:
        """The names cancelled: the one given, or every one for 'all'."""
        name = (name or "").strip()
        with self._lock:
            if name.lower() == "all":
                names = list(self._items)
            else:
                names = [name] if name in self._items else []
            for gone in names:
                del self._items[gone]
        return names

    def clear(self) -> None:
        """Drop every schedule; Stop microscope and Clear context call this."""
        with self._lock:
            self._items.clear()

    def pop_due(self) -> dict[str, Any] | None:
        """The schedule that is due first, if any is due, else None.

        A repeating schedule is set for its next time, counted from now, so a
        period is really the period plus the turn it fires. A one-off is
        removed. One at a time, so the window runs one turn per tick.
        """
        now = self.clock()
        with self._lock:
            due = sorted(
                (item for item in self._items.values() if item["next"] <= now),
                key=lambda item: item["next"],
            )
            if not due:
                return None
            item = due[0]
            if "every_seconds" in item:
                item["next"] = now + item["every_seconds"]
            else:
                del self._items[item["name"]]
            return dict(item)

    def listing(self) -> list[dict[str, Any]]:
        """Every schedule, the one due first at the top, with when it is due."""
        now = self.clock()
        with self._lock:
            items = sorted(self._items.values(), key=lambda item: item["next"])
        return [self._listed(item, now) for item in items]

    @staticmethod
    def _listed(item: dict[str, Any], now: float) -> dict[str, Any]:
        keys = ("name", "instruction", "every_seconds", "in_seconds", "at")
        listed = {key: item[key] for key in keys if key in item}
        listed["due_in_s"] = int(max(0.0, item["next"] - now))
        listed["due_at"] = time.strftime(CLOCK_FORMAT, time.localtime(item["next"]))
        return listed


def _seconds(value: Any) -> int:
    try:
        seconds = round(float(value))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"seconds must be a number, not {value!r}") from None
    if seconds < SCHEDULE_MIN_SECONDS:
        raise ValueError(f"at least {SCHEDULE_MIN_SECONDS} seconds")
    return seconds


def _clock_time(text: Any) -> str:
    """'HH:MM' or 'HH:MM:SS' (24-hour) as 'HH:MM'."""
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})(?::\d{2})?\s*", str(text))
    if not match or not (0 <= int(match.group(1)) < 24 and 0 <= int(match.group(2)) < 60):
        raise ValueError(f"a clock time is HH:MM, not {text!r}")
    return f"{int(match.group(1)):02d}:{match.group(2)}"


def _next_occurrence(at: str, now: float) -> float:
    """The next time the local clock reads ``at``: today, or tomorrow if that has passed."""
    hour, minute = (int(part) for part in at.split(":"))
    today = time.localtime(now)
    for day in (today.tm_mday, today.tm_mday + 1):  # mktime carries a day past the month's end
        candidate = time.mktime((today.tm_year, today.tm_mon, day, hour, minute, 0, 0, 0, -1))
        if candidate > now:
            return candidate
    return candidate
