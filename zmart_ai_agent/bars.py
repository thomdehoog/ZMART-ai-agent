"""The bars above the input line: the open request, and one row per schedule.

The request bar shows what the operator's last message set going (see
``requests.py``): its number, how many turns and tokens it has taken, what it
waits for, and its plan, the checklist the agent wrote, ticked off as the
steps are done. Its Cancel request button ends the request, with any wait it
had pending, without stopping the microscope. The schedule rows show each
schedule the agent has set (see ``schedules.py``): how often it fires and a
countdown to its next firing, each with a Cancel schedule button of its own.
The window refreshes both once a second, on the same tick that fires the
schedules.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-09
License: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .requests import Request

MUTED = "color:#555"


def countdown(seconds: float) -> str:
    """Seconds as m:ss, or h:mm:ss from an hour."""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def period(seconds: float) -> str:
    """A repeat period in the plainest unit: "2 h", "3 min", "45 s"."""
    seconds = int(seconds)
    if seconds % 3600 == 0:
        return f"{seconds // 3600} h"
    if seconds % 60 == 0:
        return f"{seconds // 60} min"
    return f"{seconds} s"


def schedule_text(item: dict[str, Any], running: bool) -> str:
    """One schedule row: its name, how often, and how long until it fires.

    A due schedule waits for the turn that is running, since schedules fire
    only between turns.
    """
    if "every_seconds" in item:
        how = f"every {period(item['every_seconds'])}"
    elif "at" in item:
        how = f"once at {item['at']}"
    else:
        how = "once"
    if running and item["due_in_s"] == 0:
        when = "due, after this turn"
    else:
        when = f"next in {countdown(item['due_in_s'])}"
    return f"⏱ {item['name'].replace('_', ' ')} · {how} · {when}"


def request_text(request: Request) -> str:
    """The request bar's words: the request's number, turns, tokens, wait and plan."""
    text = f"Request {request.number}: {request.turns} turns, {request.tokens:,} tokens"
    if request.wait:
        text += f", waiting {countdown(request.wait['seconds'])}"
    if request.plan:
        text += "\n" + "\n".join(request.plan)
    return text


class RequestBar(QWidget):
    """The open request on one line, with Cancel request; hidden while there is none."""

    def __init__(self, on_cancel: Callable[[], None]) -> None:
        super().__init__()
        self.label = QLabel(wordWrap=True)
        self.label.setStyleSheet(MUTED)
        self.cancel_button = QPushButton("Cancel request", clicked=on_cancel)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.label, 1)
        row.addWidget(self.cancel_button)
        self.hide()

    def show_request(self, request: Request | None) -> None:
        """Show ``request``, or nothing when there is no open request."""
        self.setVisible(request is not None)
        if request is not None:
            self.label.setText(request_text(request))


class ScheduleRows(QWidget):
    """One row per schedule, in the order they fire; none while nothing is scheduled."""

    def __init__(self, on_cancel: Callable[[str], None]) -> None:
        super().__init__()
        self.on_cancel = on_cancel
        self.rows: dict[str, tuple[QWidget, QLabel]] = {}  # name -> (row, its label)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(2)
        self.hide()

    def show_schedules(self, listing: list[dict[str, Any]], running: bool) -> None:
        """Refresh the rows from the scheduler's listing: a row per schedule, the
        countdown on each, and no rows at all when there are no schedules."""
        names = [item["name"] for item in listing]
        if names != list(self.rows):  # one added, cancelled, fired for good, or reordered
            for row, _ in self.rows.values():
                self.layout_.removeWidget(row)
                row.hide()
                row.deleteLater()
            self.rows = {name: self._row(name) for name in names}
            for row, _ in self.rows.values():
                self.layout_.addWidget(row)
        for item in listing:
            self.rows[item["name"]][1].setText(schedule_text(item, running))
        self.setVisible(bool(names))

    def _row(self, name: str) -> tuple[QWidget, QLabel]:
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        label = QLabel()
        label.setStyleSheet(MUTED)
        cancel = QPushButton("Cancel schedule", clicked=lambda: self.on_cancel(name))
        line.addWidget(label, 1)
        line.addWidget(cancel)
        return row, label
