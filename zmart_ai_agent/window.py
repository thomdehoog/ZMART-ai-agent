"""A chat window for the microscope agent.

    zmart-ai-agent                                      # the simulated microscope
    zmart-ai-agent --driver my-scope                    # an installed driver, by its name
    zmart-ai-agent --driver my-scope --connection '{"password": "..."}'

Works with any microscope that has a ZMART driver installed on this computer.
A driver is installed once with the ZMART Controller
(``zmart_controller.register_driver``), and the controller lists it by the
name in its ``zmart_driver.json``; ``mock``, the simulated microscope that
comes with the controller, is always listed and is used when none is named.
At the top, the Driver box offers those names and connects to the chosen
one. Below it, the Model panel chooses the model to talk to (a cloud model
with its API key, a server you run yourself, or a model file on this
computer). Left: the conversation, the open request and the schedules, and
the buttons. Right: the latest image, the microscope status, and a red
banner for anything refused. The divider between the two halves can be
dragged. A clock in the window fires the schedules the agent sets ("look
every three minutes") and continues a request whose wait is over, each as a
turn of its own.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import argparse
import html
import importlib
import json
import sys
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic_ai.exceptions import UnexpectedModelBehavior
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import models
from .agent import Conversation
from .bars import RequestBar, ScheduleRows
from .images import as_png
from .instructions import (
    CHOOSE_STEPS,
    CONNECT_STEPS,
    CONTINUATION_SHOWN,
    CONTINUATION_TURN,
    SCHEDULED_SHOWN,
    SCHEDULED_TURN,
)
from .local import CONTEXT_TOO_SMALL_HELP, CONTEXT_TOO_SMALL_SIGNS
from .microscope import Microscope, instruments
from .panel import ModelPanel, PreferencesBox
from .settings import DEFAULT_DRIVER, DEFAULT_PROVIDER, FONT_POINTS

# The colour of each voice in the transcript. A turn the machine wrote (a schedule
# that fell due, a wait that is over) is a muted line, so it does not look typed.
COLOURS = {"you": "#1a5fb4", "agent": "#26a269", "system": "#b00020", "machine": "#777"}

WELCOME = (
    "Hello. I can move the stage, change the microscope's settings, focus, look at the "
    "sample and run acquisitions, on any microscope with a ZMART driver. Ask me in your "
    "own words, for example <i>What do you see?</i> or <i>Take a picture at three "
    "positions</i>. Before a long stage move, a routine or an acquisition I ask you here "
    "first."
)


def choose_driver(name: str) -> Any:
    """The driver the operator named: an installed driver's name as the controller
    lists it, or else a Python module that is a driver (as the tests use).

    Raises ``ValueError`` naming the installed drivers when neither fits.
    """
    name = name.strip()
    if name in instruments():
        return name
    try:
        return importlib.import_module(name)
    except ImportError:
        raise ValueError(
            f"no driver is installed as {name!r}; installed: {', '.join(instruments())}"
        ) from None


class _Signals(QObject):
    """Carries results from the agent's thread to the window's thread."""

    reply = Signal(str)
    error = Signal(str)
    image = Signal(object, str)
    warning = Signal(str)
    tool = Signal(str, dict)


class AgentWindow(QMainWindow):
    """The chat window. ``endpoint`` is the model to start with; None keeps the
    conversation's own model (a test passes one in that way) and leaves the panel
    to the operator."""

    def __init__(
        self,
        conversation: Conversation,
        endpoint: models.Endpoint | None = None,
    ) -> None:
        super().__init__()
        self.conversation = conversation
        self.setWindowTitle("ZMART AI agent")
        self.resize(1200, 750)

        self.signals = _Signals()
        self.signals.reply.connect(self._show_reply)
        self.signals.error.connect(self._show_error)
        self.signals.image.connect(self._show_image)
        self.signals.warning.connect(self._show_warning)
        self.signals.tool.connect(self._show_tool)
        microscope = conversation.microscope
        microscope.on_image = self.signals.image.emit
        microscope.on_warning = self.signals.warning.emit
        microscope.on_tool = self.signals.tool.emit

        # top: which driver, by the name the controller lists it under; the
        # connection entries are never shown, since they may hold a password
        self.driver_box = QComboBox()
        self.driver_box.setEditable(True)
        self.driver_box.addItems(instruments())
        self.driver_box.setCurrentText(microscope.name if microscope.driver is not None else "")
        self.driver_box.lineEdit().setPlaceholderText(f"the driver, e.g. {DEFAULT_DRIVER}")
        self.driver_box.lineEdit().returnPressed.connect(self.connect_chosen)
        self.connect_button = QPushButton("Connect", clicked=self.connect_chosen)
        microscope_row = QHBoxLayout()
        microscope_row.addWidget(QLabel("Driver:"))
        microscope_row.addWidget(self.driver_box, 1)
        microscope_row.addWidget(self.connect_button)

        # left: the conversation
        self.transcript = QTextBrowser()
        self.request_bar = RequestBar(self.cancel_request)
        self.schedule_rows = ScheduleRows(self.cancel_schedule)
        self.prompt = QLineEdit(placeholderText="Ask the microscope agent ...")
        self.prompt.returnPressed.connect(self.send)
        self.send_button = QPushButton("Send", clicked=self.send)
        self.stop_button = QPushButton("Stop microscope", clicked=self.stop_microscope)
        self.stop_button.setStyleSheet("color:#b00020; font-weight:bold")
        input_row = QHBoxLayout()
        input_row.addWidget(self.prompt, 1)
        input_row.addWidget(self.send_button)
        input_row.addWidget(self.stop_button)
        self.cancel_button = QPushButton("Cancel prompt", clicked=self.cancel_prompt)
        self.clear_button = QPushButton("Clear context", clicked=self.clear_context)
        self.show_tools = QCheckBox("Show tool calls")
        buttons_row = QHBoxLayout()
        buttons_row.addWidget(self.cancel_button)
        buttons_row.addWidget(self.clear_button)
        buttons_row.addWidget(self.show_tools)
        buttons_row.addStretch(1)

        # top: which model, folding open to the choice and the preferences
        self.preferences = PreferencesBox(self.font().pointSize(), self.set_font_size)
        self.panel = ModelPanel(
            self.use_model, lambda text: self._say("system", text), self.preferences
        )
        # The window's clock: every second, the bars are refreshed, and a continuation
        # or a schedule that fell due runs as a turn.
        self._tick = QTimer(self)
        self._tick.timeout.connect(self.tick)
        self._tick.start(1000)

        left = QVBoxLayout()
        left.addLayout(microscope_row)
        left.addWidget(self.panel)
        left.addWidget(self.transcript, 1)
        left.addWidget(self.request_bar)
        left.addWidget(self.schedule_rows)
        left.addLayout(input_row)
        left.addLayout(buttons_row)

        # right: warning, image, status
        self.warning = QLabel(wordWrap=True)
        self.warning.setStyleSheet(
            "background:#b00020; color:white; padding:8px; font-weight:bold; border-radius:4px"
        )
        self.warning.hide()
        self.image = QLabel("No image yet.", alignment=Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(480, 480)
        self.image.setStyleSheet("background:#111; color:#aaa")
        self.caption = QLabel(wordWrap=True)
        self.status = QLabel(wordWrap=True)
        self.status.setStyleSheet("color:#555")
        right = QVBoxLayout()
        right.addWidget(self.warning)
        right.addWidget(self.image, 1)
        right.addWidget(self.caption)
        right.addWidget(self.status)

        # the two halves, with a divider the operator can drag
        left_widget, right_widget = QWidget(), QWidget()
        left_widget.setLayout(left)
        right_widget.setLayout(right)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)

        self._say("agent", WELCOME, escape=False)
        if microscope.session is not None:
            self._say_connected()
        elif microscope.driver is not None:
            self._connect()
        else:
            self._say_steps(CHOOSE_STEPS)
        self._refresh_status()
        if endpoint is not None:
            self.panel.language.show_endpoint(endpoint)
            self.panel.apply()

    # -- the microscope ---------------------------------------------------------------

    def connect_chosen(self) -> None:
        """Plug in the driver named in the Driver box, and connect to its microscope.

        Choosing another driver starts a new conversation: positions,
        settings and plans of one microscope mean nothing on another.
        """
        name = self.driver_box.currentText().strip()
        if self.busy or not name:
            return
        try:
            chosen = choose_driver(name)
        except ValueError as exc:
            self._say("system", str(exc))
            self._say_steps(CHOOSE_STEPS)
            return
        microscope = self.conversation.microscope
        if microscope.driver is not None and microscope.driver != chosen:
            self.conversation.clear()
            self._say("system", "Another microscope: the conversation starts afresh.")
        microscope.driver = chosen
        self._connect()
        self._refresh_status()
        self.refresh_bars()

    def _connect(self) -> None:
        microscope = self.conversation.microscope
        try:
            microscope.connect()
        except Exception:
            self._say(
                "system",
                f"{microscope.name} could not connect: {microscope.connect_error}\n"
                + "\n".join(f"{i}. {step}" for i, step in enumerate(CONNECT_STEPS, 1)),
            )
            return
        self._say_connected()

    def _say_connected(self) -> None:
        """Which microscope, where its images go, and whether its driver describes it."""
        microscope = self.conversation.microscope
        output_root = (microscope.learned.get("info") or {}).get("output_root", "(not given)")
        text = f"Connected to {microscope.name}. Images are saved by its driver in {output_root}."
        if not microscope.has_description:
            text += (
                " Its driver gives no description of the microscope, so the agent works "
                "from its readings only: it knows the settings by name, but not what they mean."
            )
        self._say("system", text)

    def _say_steps(self, steps: list[str]) -> None:
        self._say("system", "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1)))

    def set_font_size(self, points: int) -> None:
        """Bigger or smaller letters, everywhere in the window, at once."""
        app = QApplication.instance()
        font = QFont(app.font())
        font.setPointSize(points)
        app.setFont(font)
        for widget in [self, *self.findChildren(QWidget)]:
            widget.setFont(font)

    def use_model(self, endpoint: models.Endpoint, vision: models.Endpoint | None) -> None:
        """The panel's choice: talk to this model from the next message on.

        The chat is kept. The eyes start afresh when their model changes, since
        another model cannot read the looks of the old one.
        """
        seen = self.conversation.microscope.eyes.frames
        self.conversation.use(endpoint, vision)
        note = f" The eyes start afresh; their {seen} looks so far are forgotten." if seen else ""
        self._say("system", f"Talking to {endpoint.name}.{note}")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Do not close in the middle of an action: the microscope would be left mid-way."""
        if self.busy:
            QMessageBox.information(
                self,
                "Still working",
                "The agent is still working on the microscope. Wait until it is done, "
                "or press Stop microscope, then close.",
            )
            event.ignore()
            return
        self.panel.stop_servers()  # a model file served by this window ends with it
        event.accept()

    # -- one turn of the conversation ----------------------------------------------

    def send(self) -> None:
        text = self.prompt.text().strip()
        if not text or self.busy:
            return
        self.prompt.clear()
        self.warning.hide()
        self._say("you", text)
        self._in_background(lambda: self.conversation.send(text))

    def tick(self) -> None:
        """The window's clock calls this every second: the bars are refreshed, and when
        no turn is running, a continuation whose wait is over, else a schedule that
        is due, runs as a turn of its own, shown as a muted line in the transcript.
        While a turn runs they wait for the next tick.
        """
        self.refresh_bars()
        if self.busy:
            return
        microscope = self.conversation.microscope
        if (due := microscope.requests.due()) is not None:
            request, result = due
            self._say("machine", CONTINUATION_SHOWN.format(number=request.number, result=result))
            text = CONTINUATION_TURN.format(number=request.number, result=result)
            self._in_background(
                lambda: self.conversation.send(text, "continuation", request.number)
            )
            return
        item = microscope.scheduler.pop_due()
        if item is None:
            return
        shown = SCHEDULED_SHOWN.format(
            name=item["name"].replace("_", " "), instruction=item["instruction"]
        )
        text = SCHEDULED_TURN.format(name=item["name"], instruction=item["instruction"])
        self.warning.hide()
        self._say("machine", shown)
        self._in_background(
            lambda: self.conversation.send(text, "scheduled", item.get("request")), item["name"]
        )

    def refresh_bars(self) -> None:
        """The request bar and the schedule rows, as things stand now."""
        microscope = self.conversation.microscope
        self.request_bar.show_request(microscope.requests.open())
        self.schedule_rows.show_schedules(microscope.scheduler.listing(), self.busy)

    @property
    def busy(self) -> bool:
        return not self.send_button.isEnabled()

    def _in_background(self, turn: Callable[[], str], schedule: str | None = None) -> None:
        """Run one agent turn off the window's thread, so the window stays responsive.

        A turn that fails ends with its error in the transcript. When it was a
        scheduled turn, that schedule is cancelled too, or a schedule with a
        dead model would repeat the same error every period.
        """
        self._set_busy(True)

        def work() -> None:
            try:
                self.signals.reply.emit(turn())
            except Exception as exc:
                text = _explain(exc, self.conversation.endpoint)
                if schedule and self.conversation.microscope.scheduler.cancel(schedule):
                    text += f" The schedule '{schedule}' is cancelled."
                self.signals.error.emit(text)

        threading.Thread(target=work, daemon=True).start()

    def _show_reply(self, text: str) -> None:
        self._say("agent", text)
        self._set_busy(False)
        self._refresh_status()

    def _show_error(self, text: str) -> None:
        self._say("system", text)
        self._set_busy(False)

    # -- the operator's say: Cancel prompt, Cancel request, Cancel schedule, Stop, Clear ------

    def cancel_prompt(self) -> None:
        """Stop the agent, not the microscope: further tool calls in this turn do nothing.

        What the agent already started (a move, an acquisition) runs on; Stop
        microscope ends an acquisition.
        """
        if not self.busy:
            return
        self.conversation.microscope.cancel.set()
        self._say("system", "Cancelled. The agent stops after its current step.")

    def cancel_request(self) -> None:
        """End the open request, with any wait it had pending; a turn of it that is
        running is cancelled as Cancel prompt does. The microscope is not stopped."""
        self.cancel_prompt()
        self.conversation.microscope.requests.end("cancelled by the operator")
        self._say("system", "The request is cancelled; nothing of it continues.")
        self.refresh_bars()

    def cancel_schedule(self, name: str) -> None:
        """A schedule row's Cancel: that schedule only; a turn it already started runs on.
        The agent sees it gone in the next state reading."""
        if self.conversation.microscope.scheduler.cancel(name):
            self._say("system", f"The schedule '{name}' is cancelled.")
        self.refresh_bars()

    def stop_microscope(self) -> None:
        """Cancel the agent, end a running acquisition after the current image, and
        drop every schedule and the open request."""
        self.conversation.microscope.stop()
        self._say(
            "system",
            "Stop: the agent is cancelled, a running acquisition ends after the "
            "current image, and every schedule is cancelled. A single move or image "
            "already under way finishes; use the microscope's own controls to stop it sooner.",
        )
        self.refresh_bars()

    def clear_context(self) -> None:
        """Forget the conversation, in the window and in the agent's memory."""
        if self.busy:
            return
        self.conversation.clear()
        self.transcript.clear()
        self._say("agent", WELCOME, escape=False)
        self.refresh_bars()

    # -- the right-hand side ----------------------------------------------------------

    def _show_image(self, image: np.ndarray, caption: str) -> None:
        pixmap = QPixmap()
        pixmap.loadFromData(as_png(image, bin=1).data)
        self.image.setPixmap(
            pixmap.scaled(
                self.image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.caption.setText(caption)

    def _show_tool(self, name: str, args: dict) -> None:
        if self.show_tools.isChecked():
            call = html.escape(
                f"{name}({', '.join(f'{k}={json.dumps(v)}' for k, v in args.items())})"
            )
            self.transcript.append(f'<p style="color:#888; margin:0">&#8250; {call}</p>')

    def _show_warning(self, text: str) -> None:
        self.warning.setText(f"Refused: {text}")
        self.warning.show()

    def _refresh_status(self) -> None:
        microscope = self.conversation.microscope
        if microscope.session is None:
            self.status.setText(f"{microscope.name}: not connected")
            return
        try:
            where = microscope.where()
        except Exception as exc:
            self.status.setText(f"{microscope.name}: not answering ({exc})")
            return
        p = where["position_um"]
        settings = ", ".join(
            f"{name} {value:g}" if isinstance(value, float) else f"{name} {value}"
            for name, value in (where.get("settings") or {}).items()
        )
        self.status.setText(
            f"{microscope.name}  ·  x {p['x']:.1f}, y {p['y']:.1f}, z {p['z']:.1f} um  ·  "
            f"{settings}"
        )

    # -- small helpers ------------------------------------------------------------------

    def _say(self, who: str, text: str, escape: bool = True) -> None:
        colour = COLOURS[who]
        body = html.escape(text).replace("\n", "<br>") if escape else text
        if who == "machine":  # nobody typed it: a muted line where the voice would be
            self.transcript.append(f'<p style="color:{colour}"><i>{body}</i></p>')
            return
        self.transcript.append(f'<p><b style="color:{colour}">{who}</b><br>{body}</p>')

    def _set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.prompt.setEnabled(not busy)
        self.clear_button.setEnabled(not busy)
        self.connect_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.send_button.setText("Working ..." if busy else "Send")
        self.refresh_bars()


def _explain(exc: Exception, endpoint: models.Endpoint | None) -> str:
    """Turn a failure into a sentence for the operator."""
    if isinstance(exc, UnexpectedModelBehavior):  # e.g. the model declined to answer
        return f"The agent could not answer: {exc.message}"
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if any(sign in lowered for sign in ("api_key", "api key", "authentication", "401")):
        where = f" {endpoint.name}" if endpoint else ""
        advice = models.missing_key_advice(endpoint) if endpoint else "check the API key."
        return f"The agent could not reach the model{where}: {advice} ({text})"
    if any(sign in text for sign in CONTEXT_TOO_SMALL_SIGNS):
        return f"The model server refused the request. {CONTEXT_TOO_SMALL_HELP} ({text})"
    return f"Something went wrong: {text}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chat with the ZMART AI agent, on any microscope with a ZMART driver."
    )
    parser.add_argument(
        "--driver",
        default=DEFAULT_DRIVER,
        help="the microscope's ZMART driver, by the name the controller lists it under "
        f"(zmart_controller.get_instruments); without it, {DEFAULT_DRIVER}, the simulated "
        "microscope",
    )
    parser.add_argument(
        "--connection",
        type=json.loads,
        default=None,
        help="what the driver needs to connect beyond what was saved with it, as JSON, e.g. "
        '\'{"password": "..."}\'; the driver\'s README says which entries it takes',
    )
    parser.add_argument(
        "--model",
        default=None,
        help="the model to start with, e.g. google:gemini-3.5-flash-lite or "
        "openai:gpt-5-mini; without it, the Model panel's default, which can be "
        "changed in the window",
    )
    parser.add_argument(
        "--font-size", type=int, default=FONT_POINTS, help=f"letter size in points ({FONT_POINTS})"
    )
    args = parser.parse_args(argv)

    try:
        driver = choose_driver(args.driver)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    app = QApplication(sys.argv[:1])
    font = QFont(app.font())
    font.setPointSize(args.font_size)
    app.setFont(font)
    microscope = Microscope(driver, args.connection, challenge_no_tool=True)
    endpoint = (
        models.Endpoint.from_name(args.model)
        if args.model
        else models.Endpoint.from_preset(DEFAULT_PROVIDER)
    )
    window = AgentWindow(Conversation(microscope), endpoint=endpoint)
    window.show()
    try:
        return app.exec()
    finally:
        microscope.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
