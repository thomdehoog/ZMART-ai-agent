"""The operator's requests: what one typed message set going, over as many turns as it takes.

A message the operator types opens a request. A schedule the agent sets in
that turn, and a wait it asks for, come back later as turns of the same
request, written by the machine rather than typed, so that the agent keeps
the thread of what it was asked to do. The ``wait`` tool ends a turn and
leaves one continuation pending: when the time is up, the window starts the
next turn of the request with that fact. A checklist in a reply ("- [ ]
centre the sample", "- [x] focus") is the request's plan, kept and shown in
the window as it is ticked off. Stop microscope, Cancel request, Clear
context and choosing another microscope end a request.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-09
License: MIT
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from typing import Any

from .settings import CONTINUATIONS_MAX, PLAN_STEPS_MAX, SCHEDULE_MIN_SECONDS, WAIT_MAX_S

_CHECKLIST = re.compile(r"^\s*[-*]\s*\[([ xX])\]\s*(.+?)\s*$", re.MULTILINE)


class Request:
    """One request: its number, the operator's words, the turns and tokens it has
    taken, its plan, and the wait it has pending."""

    def __init__(self, number: int, prompt: str, started: float) -> None:
        self.number = number
        self.prompt = prompt
        self.started = started
        self.turns = 0
        self.tokens = 0
        self.continuations = 0
        self.plan: list[str] = []
        self.wait: dict[str, Any] | None = None  # {"seconds", "since"} while a wait is pending
        self.ended: str | None = None  # why it ended, or None while it is open

    def brief(self, now: float) -> dict[str, Any]:
        """The request as the model and the window see it."""
        out: dict[str, Any] = {
            "number": self.number,
            "prompt": self.prompt[:200],
            "turn": self.turns,
            "minutes": round((now - self.started) / 60, 1),
        }
        if self.plan:
            out["plan"] = self.plan
        if self.wait:
            out["waiting"] = {
                "seconds": self.wait["seconds"],
                "for_s": int(now - self.wait["since"]),
            }
        return out


class Requests:
    """The requests of the session: the current one, and the one with a pending
    continuation (at most one). Thread-safe: the tools run on the agent's thread,
    the window asks from its own."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self.current: Request | None = None
        self.waiting: Request | None = None
        self._known: dict[int, Request] = {}
        self._count = 0
        self._lock = threading.Lock()

    def typed(self, prompt: str) -> Request:
        """A message the operator typed: a new request."""
        with self._lock:
            self._count += 1
            self.current = Request(self._count, prompt, self.clock())
            self._known[self.current.number] = self.current
            return self.current

    def machine(self, number: int | None) -> Request:
        """A turn the machine wrote for request ``number``: a schedule that fell due,
        or a continuation. A number no request answers to (a schedule set before
        Clear context) gets a request of its own."""
        with self._lock:
            request = self._known.get(number) if number is not None else None
            if request is None:
                self._count += 1
                request = Request(self._count, "(scheduled)", self.clock())
                self._known[request.number] = request
            self.current = request
            return request

    def finish_turn(self, reply: str, tokens: int) -> None:
        """After a turn: count it, and take a checklist in the reply as the plan."""
        with self._lock:
            request = self.current
            if request is None:
                return
            request.turns += 1
            request.tokens += tokens
            steps = [
                f"[{'x' if mark.strip() else ' '}] {text}"
                for mark, text in _CHECKLIST.findall(reply or "")
            ]
            if steps:
                request.plan = steps[:PLAN_STEPS_MAX]

    def wait(self, seconds: Any) -> dict[str, Any]:
        """Leave a continuation pending for the current request, ``seconds`` from now.

        Returns the wait as set. A ValueError says why it cannot be set: no
        open request, a wait already pending, too many continuations, or a
        time outside the allowed range.
        """
        with self._lock:
            request = self.current
            if request is None or request.ended:
                raise ValueError("there is no request to continue")
            if request.wait is not None:
                raise ValueError("this turn already waits: end it now with one short sentence")
            if self.waiting is not None and self.waiting is not request:
                raise ValueError(
                    f"request {self.waiting.number} is already waiting; one wait at a time"
                )
            if request.continuations >= CONTINUATIONS_MAX:
                raise ValueError(
                    f"this request has continued {request.continuations} times; tell the "
                    "operator where it stands and let them say whether to go on"
                )
            request.wait = {"seconds": _seconds(seconds), "since": self.clock()}
            self.waiting = request
            return dict(request.wait)

    def is_waiting(self) -> bool:
        """True while the current request's turn has asked to wait: it must end now."""
        return self.current is not None and self.current.wait is not None

    def due(self) -> tuple[Request, str] | None:
        """The continuation whose time has come, as (request, what happened), or None."""
        with self._lock:
            request = self.waiting
            if request is None or request.wait is None:
                return None
            waited = self.clock() - request.wait["since"]
            if waited < request.wait["seconds"]:
                return None
            request.wait, self.waiting = None, None
            request.continuations += 1
            return request, f"waited {int(waited)} s"

    def end(self, reason: str) -> None:
        """End the open requests (Stop microscope, Cancel request, Clear context)."""
        with self._lock:
            for request in (self.current, self.waiting):
                if request is not None and request.ended is None:
                    request.ended, request.wait = reason, None
            self.waiting = None

    def open(self) -> Request | None:
        """The request the window shows: the waiting one, else the current one while open."""
        request = self.waiting or self.current
        return request if request is not None and request.ended is None else None


def _seconds(value: Any) -> int:
    try:
        seconds = round(float(value))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"seconds must be a number, not {value!r}") from None
    if seconds < SCHEDULE_MIN_SECONDS:
        raise ValueError(f"wait at least {SCHEDULE_MIN_SECONDS} seconds")
    if seconds > WAIT_MAX_S:
        raise ValueError(f"wait at most {WAIT_MAX_S} seconds; for longer, set a schedule")
    return seconds
