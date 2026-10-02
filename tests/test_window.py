"""The chat window, offscreen, with a scripted model and the mock driver behind the controller.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pytestqt")

import zmart_controller
from mock_microscope import MOCK
from PySide6.QtWidgets import QMessageBox
from test_agent import MOCK_OPS, Script, position, saved_images

from zmart_ai_agent import window as window_module
from zmart_ai_agent.agent import Conversation
from zmart_ai_agent.instructions import NO_DESCRIPTION
from zmart_ai_agent.microscope import Microscope
from zmart_ai_agent.window import AgentWindow, pick_instrument


@pytest.fixture
def open_window(qtbot, instrument):
    scopes = []

    def make(*steps, vision=None, connect=True, chosen=instrument):
        microscope = Microscope(chosen)
        scopes.append(microscope)
        if connect:
            microscope.connect()
        if vision is not None:
            microscope.vision_model = Script(vision).model()
        else:
            microscope.vision = False  # no vision model: the look and run tools skip it
        window = AgentWindow(
            Conversation(microscope, model=Script(*steps).model()), instruments=[instrument]
        )
        # A failed test may leave a turn running; the window refuses to close then
        # (with a dialog nobody can click), so the turn is ended before the close.
        qtbot.addWidget(window, before_close_func=settle)
        return window

    def settle(window):
        window.stop_microscope()
        qtbot.waitUntil(lambda: not window.busy, timeout=10000)

    yield make
    for microscope in scopes:
        microscope.disconnect()


def slowed(name, seconds):
    """The mock's ``name`` function, taking ``seconds`` longer: time to press a button."""
    original = MOCK_OPS[name]

    def slow(*args, **kwargs):
        time.sleep(seconds)
        return original(*args, **kwargs)

    return slow


def ask(qtbot, window, text):
    window.prompt.setText(text)
    window.send()
    qtbot.waitUntil(lambda: not window.busy, timeout=10000)
    return window.transcript.toPlainText()


def start(window, text):
    window.prompt.setText(text)
    window.send()


def test_a_question_and_its_answer(qtbot, open_window):
    window = open_window("The stage is at x 0 um.")
    assert "x 0.0, y 0.0, z 0.0 um" in window.status.text()
    assert "laser_power 10" in window.status.text()
    transcript = ask(qtbot, window, "where is the stage?")
    assert "where is the stage?" in transcript and "The stage is at x 0 um." in transcript


def test_the_window_says_which_microscope_and_where_images_go(qtbot, open_window):
    window = open_window()
    transcript = window.transcript.toPlainText()
    assert "Connected to mock / mock-scope / mock-api" in transcript
    output_root = window.conversation.microscope.learned["info"]["output_root"]
    assert output_root in transcript and NO_DESCRIPTION not in transcript
    assert window.instrument_box.currentText() == "mock / mock-scope / mock-api"


def test_a_driver_without_a_description_is_said_in_the_window(qtbot, open_window, monkeypatch):
    original = MOCK_OPS["get_info"]

    def get_info(handle):
        answer = original(handle)
        answer["report"].pop("description")
        return answer

    monkeypatch.setitem(MOCK_OPS, "get_info", get_info)
    window = open_window()
    assert "gives no description" in window.transcript.toPlainText()


def test_choosing_and_connecting_a_microscope(qtbot, open_window):
    window = open_window("Connected now.", connect=False, chosen=None)
    transcript = window.transcript.toPlainText()
    assert "Choose the microscope" in transcript and "not connected" in window.status.text()
    window.instrument_box.setCurrentIndex(0)
    window.connect_button.click()
    assert window.conversation.microscope.session is not None
    assert "Connected to mock / mock-scope / mock-api" in window.transcript.toPlainText()
    assert "x 0.0, y 0.0, z 0.0 um" in window.status.text()


def test_a_microscope_that_does_not_connect_says_why(qtbot, open_window, instrument):
    window = open_window(connect=False, chosen={**instrument, "mock_timing": "bogus"})
    assert "could not connect" in window.transcript.toPlainText()
    assert "mock_timing" in window.transcript.toPlainText()


def test_with_no_microscope_registered_the_window_says_how(qtbot, instrument, monkeypatch):
    monkeypatch.setattr(zmart_controller, "get_instruments", list)
    microscope = Microscope(None, vision=False)
    window = AgentWindow(Conversation(microscope, model=Script().model()))
    qtbot.addWidget(window)
    assert "No microscope is registered" in window.transcript.toPlainText()
    assert window.instrument_box.count() == 0 and not window.connect_button.isEnabled()


def test_pick_instrument_by_name_or_the_only_one():
    one = {**MOCK, "client": "secret"}
    other = {"vendor": "acme", "microscope": "a5", "api": "sdk"}
    assert pick_instrument([one], None) is one  # the only one
    assert pick_instrument([one, other], None) is None  # several: the window asks
    assert pick_instrument([one, other], "acme/a5/sdk") is other
    with pytest.raises(ValueError, match="mock/mock-scope/mock-api"):
        pick_instrument([one, other], "acme/a6/sdk")


def test_a_long_move_is_asked_about_in_the_chat(qtbot, open_window):
    long = ("move_stage", {"x": 2000})
    window = open_window(long, "Shall I move 2 mm to x = 2 mm?", long, "We are at x 2 mm.")
    transcript = ask(qtbot, window, "go to x 2 mm")
    assert "Shall I move 2 mm to x = 2 mm?" in transcript
    assert position(window.conversation.microscope)["x"] == 0.0
    assert "We are at x 2 mm." in ask(qtbot, window, "yes")
    assert "x 2000.0, y 0.0" in window.status.text()


def test_cancel_prompt_stops_the_agent(qtbot, open_window, monkeypatch):
    monkeypatch.setitem(MOCK_OPS, "set_xyz", slowed("set_xyz", 0.3))  # time to press Cancel
    window = open_window(("move_stage", {"x": 100}), ("move_stage", {"x": 200}), "Stopped.")
    window.show_tools.setChecked(True)
    start(window, "move twice")
    qtbot.waitUntil(lambda: "move_stage" in window.transcript.toPlainText(), timeout=10000)
    window.cancel_button.click()
    qtbot.waitUntil(lambda: not window.busy, timeout=10000)
    assert position(window.conversation.microscope)["x"] == 100.0  # the second move did not run
    assert "Cancelled." in window.transcript.toPlainText()


def test_tool_calls_show_when_asked(qtbot, open_window):
    window = open_window(("move_stage", {"x": 100}), "Moved.", ("get_status", {}), "Here.")
    ask(qtbot, window, "move a little")
    assert "move_stage" not in window.transcript.toPlainText()
    window.show_tools.setChecked(True)
    assert "get_status()" in ask(qtbot, window, "status?")


def test_clear_context_empties_the_chat_and_the_memory(qtbot, open_window):
    window = open_window("One.", "Two.")
    ask(qtbot, window, "good morning")
    window.clear_context()
    assert (
        "good morning" not in window.transcript.toPlainText() and window.conversation.history == []
    )


def test_a_limit_breach_shows_the_red_banner(qtbot, open_window):
    window = open_window(("move_stage", {"z": 2000}), "That is outside the limits.")
    ask(qtbot, window, "go to z 2 mm")
    assert window.warning.isVisibleTo(window) and "outside the range" in window.warning.text()
    assert position(window.conversation.microscope)["z"] == 0.0


def test_looking_fills_the_image_panel(qtbot, open_window):
    window = open_window(("look", {"question": "focus?"}), "Looks sharp.", vision="Sharp spots.")
    ask(qtbot, window, "look")
    assert window.image.pixmap() is not None and not window.image.pixmap().isNull()
    assert window.caption.text() == "focus?"


def test_the_model_panel_folds_and_applies_a_choice(qtbot, open_window, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    window = open_window()
    panel = window.panel
    assert panel.body.isHidden() and "no model" in panel.summary.text()
    panel.toggle.setChecked(True)
    assert not panel.body.isHidden()
    picker = panel.language
    picker.provider.setCurrentText("Gemini")
    assert picker.model.text() == "gemini-3.5-flash-lite" and picker.base_url.isHidden()
    # without a key, nothing changes and the panel stays open with the advice
    panel.apply()
    assert "The model was not changed" in window.transcript.toPlainText()
    assert "GEMINI_API_KEY" in panel.note.text() and not panel.body.isHidden()
    # with a key, the agent talks to that model and the panel folds shut
    picker.key.setText("secret")
    panel.apply()
    assert window.conversation.endpoint.provider == "Gemini"
    assert type(window.conversation.model).__name__ == "GoogleModel"
    assert "Gemini" in panel.summary.text() and panel.body.isHidden()
    assert "Talking to Gemini" in window.transcript.toPlainText()
    assert "secret" not in window.transcript.toPlainText()


def test_a_server_picker_shows_its_address_and_a_file_picker_its_files(
    qtbot, open_window, tmp_path
):
    window = open_window()
    picker = window.panel.vision
    assert picker.same
    picker.mode.setCurrentText("Cloud")
    picker.provider.setCurrentText("OpenAI-style server")
    assert not picker.base_url.isHidden() and not picker.sees.isHidden()
    picker.sees.setChecked(True)
    assert picker.cloud_endpoint().vision and not picker.cloud_endpoint().needs_key
    (tmp_path / "tiny-1b.gguf").write_bytes(b"")
    picker.models_folder = tmp_path
    picker.mode.setCurrentText("File on this computer")
    assert picker.local_model.currentText() == "tiny-1b.gguf" and picker.key.isHidden()
    assert picker.local_path() == tmp_path / "tiny-1b.gguf"


def test_the_preferences_change_the_letters(qtbot, open_window):
    window = open_window()
    window.panel.preferences.font_size.setValue(14)
    assert window.transcript.font().pointSize() == 14


def test_a_due_schedule_waits_for_a_running_turn(qtbot, open_window, monkeypatch):
    window = open_window(("get_status", {}), "Slow status.", "Fired.")
    monkeypatch.setitem(MOCK_OPS, "get_xyz", slowed("get_xyz", 1.0))  # a slow turn
    scheduler = window.conversation.microscope.scheduler
    scheduler.add("watch", "hello", every_seconds=60)
    scheduler.clock = lambda: time.time() + 61
    start(window, "status")  # a turn is running; the due schedule must wait
    qtbot.wait(1200)
    assert "[scheduled" not in window.transcript.toPlainText() and window.busy
    qtbot.waitUntil(lambda: "Fired." in window.transcript.toPlainText(), timeout=20000)
    transcript = window.transcript.toPlainText()
    assert transcript.index("Slow status.") < transcript.index("[scheduled")


def test_a_failing_scheduled_turn_cancels_its_schedule(qtbot, open_window):
    window = open_window()  # an empty script: the model fails on the first call
    scheduler = window.conversation.microscope.scheduler
    scheduler.add("watch", "look", every_seconds=60)
    scheduler.clock = lambda: time.time() + 61
    qtbot.waitUntil(lambda: "is cancelled" in window.transcript.toPlainText(), timeout=10000)
    assert scheduler.listing() == [] and "Something went wrong" in window.transcript.toPlainText()


def test_a_due_schedule_runs_as_its_own_turn_and_stop_drops_it(qtbot, open_window):
    set_it = ("schedule", {"name": "watch", "instruction": "status", "every_seconds": 60})
    window = open_window(set_it, "Every minute.", ("get_status", {}), "Here is the status.")
    ask(qtbot, window, "status every minute")
    scheduler = window.conversation.microscope.scheduler
    assert [s["name"] for s in scheduler.listing()] == ["watch"]
    scheduler.clock = lambda: time.time() + 61  # a minute passes
    qtbot.waitUntil(
        lambda: "[scheduled 'watch'] status" in window.transcript.toPlainText(), timeout=5000
    )
    qtbot.waitUntil(lambda: not window.busy, timeout=10000)
    assert "Here is the status." in window.transcript.toPlainText()
    window.stop_microscope()
    assert scheduler.listing() == []
    assert "every schedule is cancelled" in window.transcript.toPlainText()


def test_the_halves_sit_in_a_splitter(qtbot, open_window):
    window = open_window()
    assert window.splitter.count() == 2


def test_an_error_is_shown_and_the_window_stays_usable(qtbot, open_window):
    window = open_window()  # the script is empty: the "model" fails on the first call
    transcript = ask(qtbot, window, "hello")
    assert "Something went wrong" in transcript and window.prompt.isEnabled()


PLAN = {"name": "run", "channels": [{"name": "dim", "settings": {"laser_power": 5}}]}


def test_an_acquisition_runs_from_the_window_and_stop_ends_it(qtbot, open_window, monkeypatch):
    monkeypatch.setitem(MOCK_OPS, "acquire", slowed("acquire", 0.1))  # time to press Stop
    run = ("run_acquisition", {"plan_id": "run-1"})
    plan = {**PLAN, "time_points": 12}
    window = open_window(("plan_acquisition", plan), "Shall I start 12?", run, "Stopped early.")
    assert "Shall I start 12?" in ask(qtbot, window, "take 12 images")
    microscope = window.conversation.microscope
    assert saved_images(microscope) == []  # nothing starts before the operator agrees
    start(window, "yes")
    qtbot.waitUntil(lambda: "t002" in window.caption.text(), timeout=10000)
    window.stop_button.click()
    qtbot.waitUntil(lambda: not window.busy, timeout=10000)
    transcript = window.transcript.toPlainText()
    assert "Stop: the agent is cancelled" in transcript and "Stopped early." in transcript
    assert 3 <= len(saved_images(microscope)) < 12


def test_the_window_will_not_close_mid_action(qtbot, open_window, monkeypatch):
    told = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: told.append(a[1]))
    window = open_window("Hi.")
    window._set_busy(True)
    assert window.close() is False and told == ["Still working"]
    window._set_busy(False)
    assert window.close() is True


def test_main_plugs_in_a_driver_for_the_session_and_picks_the_instrument(monkeypatch, instrument):
    opened = {}

    class FakeApp:
        def __init__(self, argv):
            pass

        def font(self):
            from PySide6.QtGui import QFont

            return QFont()

        def setFont(self, font):
            pass

        def exec(self):
            return 0

    class FakeWindow:
        def __init__(self, conversation, endpoint=None, instruments=None):
            opened["microscope"] = conversation.microscope

        def show(self):
            pass

    registered = []
    monkeypatch.setattr(window_module, "QApplication", FakeApp)
    monkeypatch.setattr(window_module, "AgentWindow", FakeWindow)
    monkeypatch.setattr(
        zmart_controller, "register_driver", lambda path, remember: registered.append(remember)
    )
    driver = str(Path(__file__).parent)
    assert window_module.main(["--driver", driver, "--instrument", "mock/mock-scope/mock-api"]) == 0
    assert registered == [False]  # for this session only
    assert opened["microscope"].name == "mock / mock-scope / mock-api"
    assert window_module.main(["--instrument", "nope/nope/nope"]) == 2
