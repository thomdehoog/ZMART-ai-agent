"""A chat window for the microscope agent.

    zmart-ai-agent
    zmart-ai-agent --instrument mock/mock-scope/mock-api
    zmart-ai-agent --driver path\\to\\driver           # plug a driver in for this session

Works with any microscope whose ZMART driver is registered with the ZMART
Controller on this computer. At the top, the Microscope box lists them and
connects to the one chosen; with only one registered, it is chosen by itself.
Below it, the Model panel chooses the model to talk to (a cloud model with its
API key, a server you run yourself, or a model file on this computer). Left:
the conversation and the buttons. Right: the latest image, the microscope
status, and a red banner for anything refused. The divider between the two
halves can be dragged. A clock in the window fires the schedules the
agent sets ("look every three minutes") as turns of their own.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import zmart_controller
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
from .images import as_png
from .instructions import CHOOSE_STEPS, CONNECT_STEPS, REGISTER_STEPS, SCHEDULED_TURN
from .local import CONTEXT_TOO_SMALL_HELP, CONTEXT_TOO_SMALL_SIGNS
from .microscope import Microscope, identity
from .panel import ModelPanel, PreferencesBox
from .settings import DEFAULT_PROVIDER, FONT_POINTS

# The colour of each voice in the transcript.
COLOURS = {"you": "#1a5fb4", "agent": "#26a269", "system": "#b00020", "scheduled": "#8a5a00"}

WELCOME = (
    "Hello. I can move the stage, change the microscope's settings, focus, look at the "
    "sample and run acquisitions, on any microscope with a ZMART driver. Ask me in your "
    "own words, for example <i>What do you see?</i> or <i>Take a picture at three "
    "positions</i>. Before a long stage move, a routine or an acquisition I ask you here "
    "first."
)


def instrument_name(instrument: dict[str, Any]) -> str:
    return " / ".join(identity(instrument).values())


def pick_instrument(instruments: list[dict[str, Any]], wanted: str | None) -> dict | None:
    """The instrument named "vendor/microscope/api", or the only one when none is named.

    With several registered and none named, the answer is None and the window
    lets the operator choose. A name that matches none raises ValueError,
    listing the names there are.
    """
    if wanted is None:
        return instruments[0] if len(instruments) == 1 else None
    for instrument in instruments:
        if "/".join(identity(instrument).values()) == wanted.strip():
            return instrument
    known = ", ".join("/".join(identity(i).values()) for i in instruments) or "none"
    raise ValueError(f"no registered microscope is called {wanted!r}; registered: {known}")


class _Signals(QObject):
    """Carries results from the agent's thread to the window's thread."""

    reply = Signal(str)
    error = Signal(str)
    image = Signal(object, str)
    warning = Signal(str)
    tool = Signal(str, dict)


class AgentWindow(QMainWindow):
    """The chat window. ``endpoint`` is the model to start with; None keeps the
    conversation's own model (a test passes one in that way) and leaves the panel to the operator.
    ``instruments`` are the microscopes offered; by default those the controller lists."""

    def __init__(
        self,
        conversation: Conversation,
        endpoint: models.Endpoint | None = None,
        instruments: list[dict[str, Any]] | None = None,
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

        # top: which microscope; only the three names are shown, never the rest of
        # the connection dictionary, which may hold a password
        self.instruments = (
            zmart_controller.get_instruments() if instruments is None else list(instruments)
        )
        if microscope.instrument is not None and not any(
            identity(i) == identity(microscope.instrument) for i in self.instruments
        ):
            self.instruments.append(microscope.instrument)
        self.instrument_box = QComboBox()
        self.instrument_box.addItems([instrument_name(i) for i in self.instruments])
        if microscope.instrument is not None:
            self.instrument_box.setCurrentText(instrument_name(microscope.instrument))
        else:
            self.instrument_box.setCurrentIndex(-1)
        self.connect_button = QPushButton("Connect", clicked=self.connect_chosen)
        self.connect_button.setEnabled(bool(self.instruments))
        microscope_row = QHBoxLayout()
        microscope_row.addWidget(QLabel("Microscope:"))
        microscope_row.addWidget(self.instrument_box, 1)
        microscope_row.addWidget(self.connect_button)

        # left: the conversation
        self.transcript = QTextBrowser()
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
        # The window's clock: every second, a schedule that fell due runs as a turn.
        self._tick = QTimer(self)
        self._tick.timeout.connect(self.fire_due_schedule)
        self._tick.start(1000)

        left = QVBoxLayout()
        left.addLayout(microscope_row)
        left.addWidget(self.panel)
        left.addWidget(self.transcript, 1)
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
        elif microscope.instrument is not None:
            self._connect()
        elif not self.instruments:
            self._say_steps(REGISTER_STEPS)
        else:
            self._say_steps(CHOOSE_STEPS)
        self._refresh_status()
        if endpoint is not None:
            self.panel.language.show_endpoint(endpoint)
            self.panel.apply()

    # -- the microscope ---------------------------------------------------------------

    def connect_chosen(self) -> None:
        """Connect to the microscope chosen in the Microscope box.

        Choosing another microscope starts a new conversation: positions,
        settings and plans of one microscope mean nothing on another.
        """
        index = self.instrument_box.currentIndex()
        if self.busy or index < 0:
            return
        microscope = self.conversation.microscope
        chosen = self.instruments[index]
        if microscope.instrument is not None and identity(microscope.instrument) != identity(
            chosen
        ):
            self.conversation.clear()
            self._say("system", "Another microscope: the conversation starts afresh.")
        microscope.instrument = chosen
        self._connect()
        self._refresh_status()

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
        another model cannot read the images and answers of the old one.
        """
        seen = self.conversation.microscope.eyes.frames
        self.conversation.use(endpoint, vision)
        note = (
            f" The eyes start afresh; the {seen} images seen so far are forgotten." if seen else ""
        )
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

    def fire_due_schedule(self) -> None:
        """The window's clock calls this every second. When no turn is running and a
        schedule is due, its instruction runs as a turn of its own, marked as
        scheduled in the transcript; while a turn runs it waits for the next tick.
        """
        if self.busy:
            return
        item = self.conversation.microscope.scheduler.pop_due()
        if item is None:
            return
        text = SCHEDULED_TURN.format(name=item["name"], instruction=item["instruction"])
        self.warning.hide()
        self._say("scheduled", text)
        self._in_background(lambda: self.conversation.send(text, scheduled=True), item["name"])

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

    # -- the operator's say: Cancel prompt, Stop, Clear ---------------------------------

    def cancel_prompt(self) -> None:
        """Stop the agent, not the microscope: further tool calls in this turn do nothing.

        What the agent already started (a move, an acquisition) runs on; Stop
        microscope ends an acquisition.
        """
        if not self.busy:
            return
        self.conversation.microscope.cancel.set()
        self._say("system", "Cancelled. The agent stops after its current step.")

    def stop_microscope(self) -> None:
        """Cancel the agent, end a running acquisition after the current image, and
        drop every schedule."""
        self.conversation.microscope.stop()
        self._say(
            "system",
            "Stop: the agent is cancelled, a running acquisition ends after the "
            "current image, and every schedule is cancelled. A single move or image "
            "already under way finishes; use the microscope's own controls to stop it sooner.",
        )

    def clear_context(self) -> None:
        """Forget the conversation, in the window and in the agent's memory."""
        if self.busy:
            return
        self.conversation.clear()
        self.transcript.clear()
        self._say("agent", WELCOME, escape=False)

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
            state = microscope.state()
        except Exception as exc:
            self.status.setText(f"{microscope.name}: not answering ({exc})")
            return
        p = state["position_um"]
        settings = ", ".join(
            f"{name} {value:g}" if isinstance(value, float) else f"{name} {value}"
            for name, value in (state.get("settings") or {}).items()
        )
        self.status.setText(
            f"{microscope.name}  ·  x {p['x']:.1f}, y {p['y']:.1f}, z {p['z']:.1f} um  ·  "
            f"{settings}"
        )

    # -- small helpers ------------------------------------------------------------------

    def _say(self, who: str, text: str, escape: bool = True) -> None:
        colour = COLOURS[who]
        body = html.escape(text).replace("\n", "<br>") if escape else text
        self.transcript.append(f'<p><b style="color:{colour}">{who}</b><br>{body}</p>')

    def _set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.prompt.setEnabled(not busy)
        self.clear_button.setEnabled(not busy)
        self.connect_button.setEnabled(not busy and bool(self.instruments))
        self.cancel_button.setEnabled(busy)
        self.send_button.setText("Working ..." if busy else "Send")


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
        "--instrument",
        default=None,
        help="the microscope, as vendor/microscope/api; without it, the only one registered "
        "is used, or the window asks",
    )
    parser.add_argument(
        "--driver",
        action="append",
        default=[],
        help="a driver's folder, plugged in for this session only (may be given more than "
        "once); drivers registered on this computer are found by themselves",
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

    for driver in args.driver:
        zmart_controller.register_driver(driver, remember=False)
    try:
        chosen = pick_instrument(zmart_controller.get_instruments(), args.instrument)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    app = QApplication(sys.argv[:1])
    font = QFont(app.font())
    font.setPointSize(args.font_size)
    app.setFont(font)
    microscope = Microscope(chosen, challenge_no_tool=True)
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
