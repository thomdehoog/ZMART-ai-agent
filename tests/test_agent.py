"""The agent's tools and approvals, with a scripted model in place of the real one.

``Script`` plays the model: it makes the tool calls it is given, in order, so
each test controls exactly what "the model" asks for and checks what the
microscope (the mock driver, through the real ZMART controller) and the
operator see. Every check reads the microscope back through the controller.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

import json
from pathlib import Path

import numpy as np
import pytest
from mock_microscope import MOCK, mock_ops, plug_in_mock
from pydantic_ai.messages import (
    BinaryContent,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from zmart_ai_agent.agent import Conversation
from zmart_ai_agent.eyes import Eyes, last_turns
from zmart_ai_agent.images import as_png, binned, image_statistics, read_saved, saved_files
from zmart_ai_agent.instructions import (
    CALLED_NOTHING_CHALLENGE,
    GO_AHEAD_ADVICE,
    LIMIT_ADVICE,
    NO_DESCRIPTION,
    OPTIONS_ADVICE,
)
from zmart_ai_agent.microscope import Microscope
from zmart_ai_agent.models import Endpoint
from zmart_ai_agent.settings import DEFAULT_MODEL_SETTINGS, HISTORY_KEEP_TURNS


class Script:
    """A stand-in for the model: each call returns the next step.

    A step is text (the answer), a (tool name, arguments) pair (a tool call), a
    whole ModelResponse, or an exception (the model call fails, as when the API
    is overloaded).
    """

    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests = []  # what the model was sent, per call

    def __call__(self, messages, info):
        self.requests.append(messages)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, ModelResponse):
            return step
        if isinstance(step, str):
            return ModelResponse(parts=[TextPart(step)])
        name, args = step
        return ModelResponse(parts=[ToolCallPart(name, args)])

    def model(self):
        return FunctionModel(self)


@pytest.fixture
def microscope(instrument):
    scope = Microscope(instrument)
    scope.connect()
    scope.vision = False  # no vision model in most tests; those with one say so
    scope.images, scope.warnings, scope.tools = [], [], []
    scope.on_image = lambda image, caption: scope.images.append((image, caption))
    scope.on_warning = scope.warnings.append
    scope.on_tool = lambda name, args: scope.tools.append((name, args))
    yield scope
    scope.disconnect()


def talk(microscope, *steps):
    script = Script(*steps)
    return Conversation(microscope, model=script.model()), script


def tool_results(conversation):
    return [
        part.content
        for message in conversation.history
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def position(microscope):
    """Where the stage is, read back through the controller."""
    report = microscope.session.get_xyz()["report"]
    return {axis: report[axis]["value"] for axis in "xyz"}


def settings(microscope):
    """The changeable settings, read back through the controller."""
    return microscope.session.get_state()["report"]["changeable"]


def saved_images(microscope):
    """Every image file the driver saved, outside the folder of single looks."""
    root = Path(microscope.learned["info"]["output_root"])
    return sorted(
        p
        for p in [*root.rglob("*.ome.tif"), *root.rglob("*.ome.zarr")]
        if "look" not in p.parts and "_vendor_raw" not in p.parts
    )


MOCK_OPS = mock_ops() if plug_in_mock() else {}  # the driver's functions, to replace


# -- connecting: the agent learns the microscope from the controller -----------------


def test_connecting_learns_the_microscope_from_the_controller(microscope):
    section = microscope.instrument_section()
    assert "mock / mock-scope / mock-api" in section
    assert "A pretend widefield fluorescence microscope" in section  # get_info's description
    assert "3 is 40x/0.95 Air" in section
    assert "x: from -5000 to 5000 um" in section and "motors: motoric, piezo" in section
    assert '"laser_power": 10.0' in section and '"objective": "10x/0.30 Air"' in section
    assert "z_planes" in section and "ome-zarr" in section  # get_acquisition_options
    assert "autofocus: Take a short z-stack" in section  # get_procedures, with its description
    assert microscope.learned["info"]["output_root"] in section
    assert "client" not in section  # nothing from the connection dict beyond its name


def test_the_model_is_told_this_microscope_and_the_generic_rules(microscope):
    conversation, script = talk(microscope, "Hello!")
    conversation.send("hi")
    told = script.requests[0][-1].instructions
    assert "biologist" in told and '"success"' in told  # the generic part
    assert "A pretend widefield fluorescence microscope" in told  # this microscope
    assert "right is +x" in told and "down is +y" in told  # the image-aligned frame


def test_a_driver_without_a_description_still_connects_and_says_so(instrument, monkeypatch):
    def get_info(handle):
        answer = original(handle)
        answer["report"].pop("description")
        return answer

    original = MOCK_OPS["get_info"]
    monkeypatch.setitem(MOCK_OPS, "get_info", get_info)
    scope = Microscope(instrument)
    try:
        scope.connect()
        section = scope.instrument_section()
        assert NO_DESCRIPTION in section and "A pretend widefield" not in section
        assert '"laser_power": 10.0' in section and "autofocus" in section  # still learned
        assert scope.has_description is False
    finally:
        scope.disconnect()


def test_without_a_microscope_the_model_is_told_and_check_setup_lists_them(instrument):
    scope = Microscope(None)
    conversation, script = talk(scope, ("check_setup", {}), "Choose a microscope first.")
    assert conversation.send("where is the stage?") == "Choose a microscope first."
    prompt = script.requests[0][-1].parts[-1].content
    assert "not connected" in prompt and "no microscope is chosen" in prompt
    assert "No microscope is connected" in script.requests[0][-1].instructions
    result = tool_results(conversation)[0]
    assert result["connected"] is False and MOCK in result["instruments_registered"]
    assert all("client" not in i for i in result["instruments_registered"])  # names only
    assert "steps_for_the_operator" in result


def test_a_connect_error_is_reported_by_check_setup(instrument):
    scope = Microscope({**instrument, "mock_timing": "bogus"})  # the mock refuses to connect
    conversation, _ = talk(scope, ("check_setup", {}), "It did not connect.")
    conversation.send("is the microscope there?")
    result = tool_results(conversation)[0]
    assert result["connected"] is False and "mock_timing" in result["error"]
    scope.instrument = instrument  # fixed: check_setup connects and learns it
    conversation, _ = talk(scope, ("check_setup", {}), "Connected.")
    conversation.send("try again")
    result = tool_results(conversation)[0]
    assert result["connected"] is True and result["description"] is True
    assert scope.session is not None and "autofocus" in scope.instrument_section()
    scope.disconnect()


# -- reading and small actions -----------------------------------------------------


def test_every_message_carries_the_microscope_state(microscope):
    conversation, script = talk(microscope, "Hello!")
    assert conversation.send("hi") == "Hello!"
    prompt = script.requests[0][-1].parts[-1].content
    assert prompt.startswith("hi") and "<microscope_state>" in prompt
    state = json.loads(prompt.split("<microscope_state>")[1].split("</microscope_state>")[0])
    assert state["position_um"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert state["settings"]["laser_power"] == 10.0 and state["observed"]["serial"] == "MOCK-0001"


def test_a_quoted_state_block_is_taken_out_of_the_reply(microscope):
    quoted = 'Exposure set.\n\n<microscope_state>{"position_um": {"x": 1}}</microscope_state>'
    cut_off = 'Done. <microscope_state>{"position_um": {"x": 1'  # a block the model did not close
    conversation, _ = talk(microscope, quoted, cut_off)
    assert conversation.send("50 ms please") == "Exposure set."
    assert conversation.send("thanks") == "Done."
    assert "<microscope_state>" in conversation.history[1].parts[0].content  # its own copy stays


def test_the_model_acts_one_step_at_a_time():
    assert DEFAULT_MODEL_SETTINGS["parallel_tool_calls"] is False
    assert DEFAULT_MODEL_SETTINGS["max_tokens"] >= 16000


def test_status(microscope):
    conversation, _ = talk(microscope, ("get_status", {}), "Here is the status.")
    conversation.send("where are we?")
    status = tool_results(conversation)[0]
    # The controller's answer, passed on as it is: the position, the motor, the
    # unit, how far the stage travels, and how far a picture reaches beyond that.
    x = status["position"]["x"]
    assert {k: x[k] for k in ("value", "actuator", "unit", "range")} == {
        "value": 0.0,
        "actuator": "motoric",
        "unit": "um",
        "range": [-5000.0, 5000.0],
    }
    assert x["reach"][0] <= -5000.0 and x["reach"][1] >= 5000.0
    assert status["state"]["changeable"]["exposure_ms"] == 10.0
    assert status["state"]["observed"]["objective"] == "10x/0.30 Air"


def test_the_window_hears_of_each_tool_call(microscope):
    conversation, _ = talk(microscope, ("move_stage", {"x": 100}), "Moved.")
    conversation.send("move a little")
    assert microscope.tools == [("move_stage", {"x": 100})]


def test_small_move_runs_at_once(microscope):
    conversation, _ = talk(microscope, ("move_stage", {"x": 100, "z": 10}), "Moved.")
    conversation.send("move a little")
    assert position(microscope) == {"x": 100.0, "y": 0.0, "z": 10.0}
    assert tool_results(conversation)[0]["position"] == {"x": 100.0, "y": 0.0, "z": 10.0}


def test_a_move_with_another_motor(microscope):
    call = {"z": 50, "actuators": {"z": "piezo"}}
    conversation, _ = talk(microscope, ("move_stage", call), "Moved with the piezo.")
    conversation.send("move the piezo up 50")
    assert position(microscope)["z"] == 50.0
    assert tool_results(conversation)[0]["actuators"]["z"] == "piezo"


def test_limit_breach_is_refused_with_advice_and_shown_in_the_window(microscope):
    conversation, _ = talk(microscope, ("move_stage", {"z": 2000}), "That is outside the limits.")
    conversation.send("go to z 2 mm")
    error = tool_results(conversation)[0]["error"]
    assert error["code"] == "limit" and error["advice"] == LIMIT_ADVICE
    assert error["message"].startswith("z = 2000 um is outside the range [-500, 500] um")
    assert microscope.warnings == [error["message"]]
    assert position(microscope) == {"x": 0.0, "y": 0.0, "z": 0.0}


def test_a_move_the_driver_refuses_is_a_refusal_and_the_stage_stays(microscope):
    call = {"z": 10, "actuators": {"z": "hydraulic"}}  # a motor this microscope does not have
    conversation, _ = talk(microscope, ("move_stage", call), "There is no such motor.")
    conversation.send("move z with the hydraulic drive")
    error = tool_results(conversation)[0]["error"]
    assert error["code"] == "limit" and "unknown actuator 'hydraulic'" in error["message"]
    assert microscope.warnings and position(microscope)["z"] == 0.0


# -- long moves are asked about in the chat first ------------------------------------------

LONG = ("move_stage", {"x": 2000})


def test_a_long_move_is_asked_about_in_the_chat_first(microscope):
    conversation, _ = talk(
        microscope, LONG, "Shall I move 2 mm to x = 2 mm?", LONG, "We are there."
    )
    assert conversation.send("go to x 2 mm") == "Shall I move 2 mm to x = 2 mm?"
    question = tool_results(conversation)[0]
    assert question["status"] == "needs_go_ahead" and question["advice"] == GO_AHEAD_ADVICE
    assert "2000 um in XY" in question["not_done_yet"] and position(microscope)["x"] == 0.0

    assert conversation.send("yes, go ahead") == "We are there."
    assert position(microscope)["x"] == 2000.0


def test_when_the_operator_says_no_nothing_moves(microscope):
    conversation, _ = talk(microscope, LONG, "Shall I?", "OK, we stay here.")
    conversation.send("go to x 2 mm")
    assert conversation.send("no, stay") == "OK, we stay here." and position(microscope)["x"] == 0.0


def test_asking_twice_in_one_turn_is_not_a_go_ahead(microscope):
    conversation, _ = talk(microscope, LONG, LONG, "Shall I?")
    conversation.send("go to x 2 mm")
    assert [r["status"] for r in tool_results(conversation)] == ["needs_go_ahead"] * 2
    assert position(microscope)["x"] == 0.0


def test_a_go_ahead_counts_only_for_the_next_message(microscope):
    conversation, _ = talk(microscope, LONG, "Shall I?", "Sure.", LONG, "Shall I?")
    conversation.send("go to x 2 mm")
    conversation.send("hmm, tell me something else first")
    conversation.send("do it")  # two messages later: asked again, not moved
    assert tool_results(conversation)[-1]["status"] == "needs_go_ahead"
    assert position(microscope)["x"] == 0.0


def test_small_steps_that_add_up_are_asked_about_too(microscope):
    steps = [("move_stage", {"z": 60 * n}) for n in (1, 2)]
    conversation, _ = talk(microscope, *steps, "Shall I go on?")
    conversation.send("walk the focus up")
    # 60 ran (60 um from where the stage was); 120 is 120 um from there, so it asks
    assert position(microscope)["z"] == 60.0
    assert "120 um in Z" in tool_results(conversation)[1]["not_done_yet"]


# -- settings ------------------------------------------------------------------------------


def test_settings_change_without_asking(microscope):
    call = {"settings": {"exposure_ms": 50, "objective": 2}}
    conversation, _ = talk(microscope, ("set_microscope", call), "Set.")
    conversation.send("the 20x at 50 ms")
    result = tool_results(conversation)[0]
    assert result["success"] is True and result["report"]["applied"] == {
        "objective": 2,
        "exposure_ms": 50,
    }
    assert settings(microscope)["exposure_ms"] == 50.0 and settings(microscope)["objective"] == 2


def test_a_setting_the_driver_does_not_know_is_refused_with_its_own_names(microscope):
    call = {"settings": {"exposure": 50, "gain": 200}}
    conversation, _ = talk(microscope, ("set_microscope", call), "Which setting?")
    conversation.send("exposure 50")
    error = tool_results(conversation)[0]["error"]
    assert error["code"] == "invalid" and "'exposure'" in error["message"]
    assert error["configured_options"] == ["laser_power", "gain", "exposure_ms", "objective"]
    assert error["advice"] == OPTIONS_ADVICE and microscope.warnings
    assert settings(microscope)["gain"] == 100.0  # nothing changed, the known one neither


def test_a_value_the_driver_refuses_is_a_refusal_with_its_reason(microscope):
    call = {"settings": {"laser_power": 90}}
    conversation, _ = talk(microscope, ("set_microscope", call), "That is above the limit.")
    conversation.send("laser to 90 percent")
    error = tool_results(conversation)[0]["error"]
    assert error["code"] == "invalid" and "outside the limits [0.0, 50.0]" in error["message"]
    assert microscope.warnings and settings(microscope)["laser_power"] == 10.0


def test_a_driver_failure_becomes_a_failure_with_advice(microscope):
    microscope.session.disconnect()  # the driver now answers every call with RuntimeError
    conversation, _ = talk(microscope, ("get_status", {}), "The microscope does not answer.")
    assert conversation.send("status?") == "The microscope does not answer."
    error = tool_results(conversation)[0]["error"]
    assert error["message"] == "RuntimeError: session is disconnected"
    assert error["code"] == "failed" and "propose one fix as a question" in error["advice"]
    assert microscope.warnings == []  # a failure, not a refusal: no red banner


# -- cancelling and failing ------------------------------------------------------------------


def test_after_cancel_no_tool_does_anything(microscope):
    def cancel_after_the_first_call(name, args):
        microscope.cancel.set()  # the operator presses Cancel during the first move

    microscope.on_tool = cancel_after_the_first_call
    steps = [("move_stage", {"x": 100}), ("move_stage", {"x": 200}), "Stopped."]
    conversation, _ = talk(microscope, *steps, ("move_stage", {"x": 300}), "Moved.")
    conversation.send("move twice")
    assert position(microscope)["x"] == 100.0
    assert tool_results(conversation)[1]["status"] == "cancelled"
    microscope.on_tool = lambda name, args: None
    conversation.send("move again")  # a new message starts without the cancel
    assert position(microscope)["x"] == 300.0


def test_the_conversation_survives_a_model_failure_after_a_move(microscope):
    overloaded = RuntimeError("API overloaded (529)")
    step = ("move_stage", {"x": 500})
    conversation, _ = talk(microscope, step, overloaded, "We are at x 0.5 mm.")
    with pytest.raises(RuntimeError, match="overloaded"):
        conversation.send("go to x 0.5 mm")
    assert position(microscope)["x"] == 500.0  # the move did happen
    assert conversation.send("are we there?") == "We are at x 0.5 mm."  # and the talk goes on
    assert tool_results(conversation)[0]["position"]["x"] == 500.0  # the model got the result


# -- focus and procedures -------------------------------------------------------------


def test_focus_runs_the_drivers_focus_procedure_at_once(microscope):
    microscope.session.set_xyz(0, 0, 6)
    call = {"entries": {"range_um": 10, "step_um": 1}}
    conversation, _ = talk(microscope, ("focus", call), "In focus.")
    conversation.send("focus")
    result = tool_results(conversation)[0]
    assert result["procedure"] == "autofocus" and result["z_before_um"] == 6.0
    assert result["z_after_um"] == position(microscope)["z"] != 6.0
    assert result["report"]["ran"] == "autofocus" and len(result["report"]["scores"]) == 11


def test_focus_without_a_focus_procedure_is_refused_plainly(microscope, monkeypatch):
    def get_procedures(handle):
        return {"success": True, "report": {"backlash_takeup": {"description": "takes up play"}}}

    monkeypatch.setitem(MOCK_OPS, "get_procedures", get_procedures)
    conversation, _ = talk(microscope, ("focus", {}), "This microscope has no autofocus.")
    conversation.send("focus please")
    error = tool_results(conversation)[0]["error"]
    assert "lists no focus procedure" in error["message"]
    assert error["configured_options"] == ["backlash_takeup"]
    assert position(microscope)["z"] == 0.0


PIEZO = ("run_procedure", {"name": "zero_piezo"})


def test_a_procedure_is_asked_about_first_then_run(microscope):
    conversation, _ = talk(microscope, PIEZO, "Shall I park the piezo?", PIEZO, "Parked.")
    conversation.send("park the piezo")
    question = tool_results(conversation)[0]
    assert question["status"] == "needs_go_ahead" and "zero_piezo" in question["not_done_yet"]
    conversation.send("yes")
    assert tool_results(conversation)[-1]["report"]["ran"] == "zero_piezo"


def test_an_unknown_procedure_is_refused_with_the_drivers_names(microscope):
    call = ("run_procedure", {"name": "calibrate"})
    conversation, _ = talk(microscope, call, "There is no such routine.")
    conversation.send("calibrate it")
    error = tool_results(conversation)[0]["error"]
    assert error["code"] == "invalid" and error["advice"] == OPTIONS_ADVICE
    assert error["configured_options"] == ["autofocus", "backlash_takeup", "zero_piezo"]


# -- looking and the eyes --------------------------------------------------------------


def test_look_acquires_reads_the_file_and_asks_a_vision_model(microscope):
    vision = Script("Bright round spots on a dark field.")
    microscope.vision_model, microscope.vision = vision.model(), True
    conversation, _ = talk(microscope, ("look", {"question": "what do you see?"}), "Spots.")
    conversation.send("look at the sample")

    result = tool_results(conversation)[0]
    assert result["answer"] == "Bright round spots on a dark field."
    assert result["statistics"]["max"] > result["statistics"]["mean"] > 0
    (saved,) = result["files"]
    assert Path(saved).is_file() and "look" in Path(saved).parts
    image, caption = microscope.images[0]
    assert image.shape == (64, 64) and caption == "what do you see?"
    # the vision model got the question, the measurements and a PNG
    question, png = vision.requests[0][-1].parts[-1].content
    assert "what do you see?" in question and "sharpness" in question
    assert png.media_type == "image/png"


def test_the_eyes_remember_earlier_images(microscope):
    vision = Script("Three spots.", "The same three spots as in image 1; nothing moved.")
    microscope.vision_model, microscope.vision = vision.model(), True
    look = ("look", {"question": "what do you see?"})
    again = ("look", {"question": "has anything changed since the image before?"})
    conversation, _ = talk(microscope, look, again, "Nothing moved.")
    conversation.send("look twice and compare")
    first, second = tool_results(conversation)
    assert first["images_seen"] == 1 and second["images_seen"] == 2
    assert second["answer"].startswith("The same three spots")
    history = vision.requests[1]
    texts = [
        p.content for m in history for p in m.parts if isinstance(p, (UserPromptPart, TextPart))
    ]
    assert any(t == "Three spots." for t in texts)
    prompt = history[-1].parts[-1].content[0]
    assert prompt.startswith("Image 2,") and "position_um" in prompt and "laser_power" in prompt


def test_ask_eyes_asks_about_the_images_seen_without_a_new_one(microscope):
    vision = Script("Three spots.", "Still three; nothing has moved.")
    microscope.vision_model, microscope.vision = vision.model(), True
    steps = [
        ("look", {"question": "what do you see?"}),
        ("ask_eyes", {"question": "has the sample moved?"}),
        "No.",
    ]
    conversation, _ = talk(microscope, *steps)
    conversation.send("look, then tell me whether it moved")
    asked = tool_results(conversation)[1]
    assert asked["answer"] == "Still three; nothing has moved." and asked["images_seen"] == 1
    assert len(microscope.images) == 1  # no new image for the question
    assert vision.requests[1][-1].parts[-1].content.startswith("No new image.")


def test_ask_eyes_before_any_look_says_so(microscope):
    vision = Script()
    microscope.vision_model, microscope.vision = vision.model(), True
    conversation, _ = talk(microscope, ("ask_eyes", {"question": "anything?"}), "Look first.")
    conversation.send("what did you see?")
    assert "look first" in tool_results(conversation)[0]["answer"] and vision.requests == []


def test_older_images_are_detached_but_their_words_stay(microscope):
    vision = Script("One.", "Two.", "Three.")
    microscope.vision_model, microscope.vision = vision.model(), True
    microscope.eyes = Eyes(vision.model(), frames_kept=1)
    look = ("look", {"question": "what?"})
    conversation, _ = talk(microscope, look, look, look, "Done.")
    conversation.send("look three times")
    turns = [m for m in microscope.eyes._history if isinstance(m.parts[0], UserPromptPart)]
    assert len(turns) == 3
    with_image = [any(isinstance(c, BinaryContent) for c in t.parts[0].content) for t in turns]
    assert with_image == [False, False, True]  # only the newest keeps its picture
    assert "[image no longer attached]" in turns[0].parts[0].content
    assert "Image 1," in turns[0].parts[0].content[0]  # the words stay


def test_the_eyes_keep_only_the_last_so_many_looks(microscope):
    vision = Script("One.", "Two.", "Three.")
    microscope.vision_model, microscope.vision = vision.model(), True
    microscope.eyes = Eyes(vision.model(), turns_kept=2)
    look = ("look", {"question": "what?"})
    conversation, _ = talk(microscope, look, look, look, "Done.")
    conversation.send("look three times")
    turns = [m for m in microscope.eyes._history if isinstance(m.parts[0], UserPromptPart)]
    assert len(turns) == 2 and "Image 2," in turns[0].parts[0].content[0]
    assert last_turns([], 3) == []


def test_a_failing_vision_model_is_reported_and_the_image_not_counted(microscope):
    vision = Script(RuntimeError("the vision model is down"))
    microscope.vision_model, microscope.vision = vision.model(), True
    conversation, _ = talk(microscope, ("look", {"question": "what?"}), "Sorry.")
    conversation.send("look")
    result = tool_results(conversation)[0]
    assert result["error"]["code"] == "failed"
    assert "vision model is down" in result["error"]["message"]
    assert microscope.eyes.frames == 0 and len(microscope.images) == 1
    # the last image of a run: the run is not turned into a failure by the describing
    microscope.vision_model = Script(RuntimeError("still down")).model()
    steps = [("plan_acquisition", PLAN), RUN, "Start?", RUN, "Saved."]
    conversation, _ = talk(microscope, *steps)
    conversation.send("take a stack at a")
    conversation.send("yes")
    run = tool_results(conversation)[-1]
    assert run["finished"] == "completed" and "still down" in run["last_image"]["vision_error"]


def test_ask_eyes_with_a_model_that_cannot_see(microscope):
    conversation, _ = talk(microscope, ("ask_eyes", {"question": "anything?"}), "No.")
    conversation.send("what did you see?")
    assert "cannot see" in tool_results(conversation)[0]["note"]


def test_changing_the_vision_model_gives_new_eyes(microscope):
    first, second = Script("One."), Script("Two.")
    microscope.vision_model, microscope.vision = first.model(), True
    conversation, _ = talk(microscope, ("look", {"question": "what?"}), "Done.")
    conversation.send("look")
    assert microscope.eyes.frames == 1
    microscope.vision_model = second.model()  # as Conversation.use does
    assert microscope.eyes.frames == 0 and microscope.eyes.model is microscope.vision_model


def test_a_model_that_cannot_see_gets_the_numbers_only(microscope):
    conversation, _ = talk(microscope, ("look", {"question": "what do you see?"}), "Dark.")
    conversation.send("look")
    result = tool_results(conversation)[0]
    assert "answer" not in result and "cannot see" in result["note"]
    assert result["statistics"]["max"] > 0 and len(microscope.images) == 1


def test_a_failed_acquisition_is_reported_softly(microscope, monkeypatch):
    def acquire(handle, **kwargs):
        return {"success": False, "report": {"confirmed": False, "reason": "no image came back"}}

    monkeypatch.setitem(MOCK_OPS, "acquire", acquire)
    conversation, _ = talk(microscope, ("look", {"question": "what?"}), "No image came back.")
    conversation.send("look")
    result = tool_results(conversation)[0]
    assert result["success"] is False and result["report"]["reason"] == "no image came back"
    assert "propose one fix" in result["advice"] and microscope.images == []


def test_binning_averages_blocks():
    image = np.arange(16, dtype=np.float64).reshape(4, 4)
    small = binned(image, 2)
    assert small.shape == (2, 2) and small[0, 0] == np.mean([0, 1, 4, 5])
    assert binned(image[:3, :3], 2).shape == (1, 1)  # the ragged edge is dropped
    assert binned(np.zeros((4, 4, 3)), 2).shape == (2, 2, 3)  # colour keeps its planes


@pytest.mark.parametrize("shape", [(2000, 1000), (300, 200, 3)], ids=["mono", "colour"])
def test_image_statistics_and_png(shape):
    image = np.zeros(shape, dtype=np.uint16)
    image[:10, :10] = 65535
    stats = image_statistics(image)
    assert stats["max"] == 65535 and stats["saturated_percent"] > 0
    assert as_png(image).media_type == "image/png"


def test_saved_ome_tiff_and_ome_zarr_files_are_read(microscope):
    tiff = microscope.session.acquire(acquisition_type="t", position_label="a")["report"]
    options = {"format": "ome-zarr", "z_planes": 3}
    zarr = microscope.session.acquire(acquisition_type="t", position_label="b", options=options)
    # The driver lists everything it saved, its own record of the capture too;
    # the agent picks the pictures out of that list, as its tools do.
    (folder,) = saved_files(zarr["report"])
    stack = read_saved([folder])
    assert stack.shape == (3, 64, 64) and stack.dtype == np.uint16
    piece = np.frombuffer((Path(folder) / "0" / "1" / "0" / "0").read_bytes(), "<u2")
    assert np.array_equal(stack[1], piece.reshape(64, 64))  # plane 1 is the piece the mock wrote
    single = read_saved(saved_files(tiff))
    assert single.shape == (64, 64) and single.max() > single.mean() > 0
    planes = saved_files(
        microscope.session.acquire(
            acquisition_type="t", position_label="c", options={"z_planes": 2}
        )["report"]
    )
    assert read_saved(planes).shape == (2, 64, 64)  # one OME-TIFF per plane, stacked
    with pytest.raises(ValueError, match="cannot read"):
        read_saved([str(Path(planes[0]).with_suffix(".png"))])


# -- schedules ---------------------------------------------------------------------------


def test_a_schedule_is_set_and_shows_in_the_state(microscope):
    steps = [
        ("schedule", {"name": "watch", "instruction": "look", "every_seconds": 180}),
        "Every three minutes.",
        "Hello.",
    ]
    conversation, script = talk(microscope, *steps)
    conversation.send("look every three minutes")
    result = tool_results(conversation)[0]
    assert result["scheduled"]["name"] == "watch" and result["scheduled"]["every_seconds"] == 180
    assert "watch" in [s["name"] for s in microscope.scheduler.listing()]
    conversation.send("hi")
    content = script.requests[-1][-1].parts[-1].content
    state = json.loads(content.split("<microscope_state>")[1][: -len("</microscope_state>")])
    assert state["schedules"][0]["name"] == "watch" and len(state["clock"]) == 8


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"name": "x", "instruction": "look", "every_seconds": 1}, "at least"),
        ({"name": "x", "instruction": "look"}, "exactly one of"),
        ({"name": "x", "instruction": "look", "at": "25:99"}, "HH:MM"),
    ],
)
def test_a_bad_schedule_is_refused_with_the_reason(microscope, args, message):
    conversation, _ = talk(microscope, ("schedule", args), "Refused.")
    conversation.send("later")
    error = tool_results(conversation)[0]["error"]
    assert error["code"] == "invalid" and message in error["message"]
    assert microscope.scheduler.listing() == []


def test_cancel_schedule_by_name_and_an_unknown_name_lists_the_known(microscope):
    microscope.scheduler.add("watch", "look", every_seconds=60)
    conversation, _ = talk(microscope, ("cancel_schedule", {"name": "nope"}), "Which one?")
    conversation.send("cancel it")
    error = tool_results(conversation)[0]["error"]
    assert error["code"] == "invalid" and error["configured_options"] == ["watch"]
    conversation, _ = talk(microscope, ("cancel_schedule", {"name": "watch"}), "Cancelled.")
    conversation.send("cancel the watch")
    assert tool_results(conversation)[0]["cancelled"] == ["watch"]
    assert not microscope.scheduler.listing()


def test_a_scheduled_turn_is_not_the_operators(microscope):
    # A scheduled move does not move the anchor, so repeated small steps still add up
    # to a question; and it does not count as the reply to a pending question.
    conversation, _ = talk(microscope, ("move_stage", {"x": 600}), "Moved.")
    conversation.send("move x to 600")
    step = ("move_stage", {"x": 1200})
    conversation, _ = talk(microscope, step, "Shall I?", step, "Shall I?")
    conversation.send("[scheduled 'creep'] move x 600 further", scheduled=True)
    assert tool_results(conversation)[-1]["status"] == "needs_go_ahead"
    conversation.send("[scheduled 'creep'] move x 600 further", scheduled=True)
    assert tool_results(conversation)[-1]["status"] == "needs_go_ahead"
    assert position(microscope)["x"] == 600.0
    assert microscope.turn == 1  # scheduled turns do not count as the operator's


def test_stop_and_clear_drop_every_schedule(microscope):
    microscope.scheduler.add("watch", "look", every_seconds=60)
    microscope.stop()
    assert microscope.scheduler.listing() == []
    microscope.scheduler.add("watch", "look", every_seconds=60)
    Conversation(microscope).clear()
    assert microscope.scheduler.listing() == []


# -- guards on the reply ---------------------------------------------------------------


def test_an_empty_reply_is_handed_back_once(microscope):
    conversation, script = talk(microscope, "_", "The stage is at x 0.")
    assert conversation.send("where?") == "The stage is at x 0."
    challenge = script.requests[1][-1].parts[-1].content
    assert "empty" in challenge
    conversation, _ = talk(microscope, "_", "...")  # empty twice: a plain fallback line
    assert "no answer in words" in conversation.send("where?")


def test_a_reply_that_called_nothing_is_challenged_when_asked(microscope):
    microscope.challenge_no_tool = True
    # the model claims to have acted; challenged, it does act, and its new reply is the answer
    conversation, script = talk(
        microscope, "I moved the stage.", ("get_status", {}), "Here is the status."
    )
    assert conversation.send("status") == "Here is the status."
    assert "No tool was called" in script.requests[1][-1].parts[-1].content
    # challenged and still nothing to call: the first reply reaches the operator as it was
    conversation, _ = talk(microscope, "Hello, how can I help?", "SAME")
    assert conversation.send("hi") == "Hello, how can I help?"
    # the "SAME" exchange is not kept: the next turn's model sees only the first reply
    conversation, script = talk(microscope, "Hello.", "SAME", "You can image.", "SAME")
    assert conversation.send("hi") == "Hello."
    assert [type(m).__name__ for m in conversation.history] == ["ModelRequest", "ModelResponse"]
    assert conversation.send("what can I do?") == "You can image."
    seen = [part for message in script.requests[2] for part in message.parts]
    assert not any(isinstance(p, RetryPromptPart) for p in seen)
    assert "SAME" not in str([getattr(p, "content", "") for p in seen])
    # a reply that opens with the guard's word, or echoes the challenge, is not passed on
    conversation, _ = talk(microscope, "SAME\nHello.", "SAME")
    assert conversation.send("hi") == "Hello."
    conversation, _ = talk(microscope, "SAME", "Hello.", "SAME")
    assert conversation.send("hi") == "Hello."
    echoed = "Validation feedback:\n" + CALLED_NOTHING_CHALLENGE
    conversation, _ = talk(microscope, echoed, "Hi.", "SAME")
    assert conversation.send("hi") == "Hi."
    # a challenge that led to a tool call stays in the history
    conversation, _ = talk(microscope, "I moved it.", ("get_status", {}), "Here is the status.")
    conversation.send("status")
    assert any(isinstance(p, RetryPromptPart) for m in conversation.history for p in m.parts)
    # off by default: no challenge, one request
    microscope.challenge_no_tool = False
    conversation, script = talk(microscope, "Hello.")
    assert conversation.send("hi") == "Hello." and len(script.requests) == 1


def test_use_switches_the_model_and_its_settings(microscope):
    conversation = Conversation(microscope)
    assert conversation.model_settings is DEFAULT_MODEL_SETTINGS
    conversation.use(Endpoint.from_preset("Gemini", api_key="k"))
    assert type(conversation.model).__name__ == "GoogleModel"
    assert conversation.model_settings["temperature"] == 0.0 and microscope.vision
    assert microscope.vision_model is conversation.model
    server = Endpoint.from_preset("OpenAI-style server")
    conversation.use(Endpoint.from_preset("OpenAI", api_key="k"), vision=server)
    assert type(microscope.vision_model).__name__ == "OpenAIChatModel" and not microscope.vision


# -- planning and running an acquisition ----------------------------------------------


PLAN = {  # the flat form, as the model sends it
    "name": "stack_test",
    "positions": [{"x": 100, "y": 200, "z": 0, "name": "a"}],
    "channels": [
        {"name": "dim", "settings": {"laser_power": 5}},
        {"name": "bright", "settings": {"laser_power": 20, "exposure_ms": 20}},
    ],
    "options": {"z_planes": 3, "z_step_um": 1},
}
RUN = ("run_acquisition", {"plan_id": "stack_test-1"})


def test_an_acquisition_starts_only_after_the_operator_saw_the_plan(microscope):
    microscope.vision_model, microscope.vision = Script("Bright spots.").model(), True
    steps = [("plan_acquisition", PLAN), RUN, "Shall I start these 2?", RUN, "Saved."]
    conversation, _ = talk(microscope, *steps)
    assert conversation.send("take a two-channel stack at a") == "Shall I start these 2?"
    plan, question = tool_results(conversation)
    assert plan["plan_id"] == "stack_test-1" and plan["acquisitions"] == 2
    summary = plan["summary"]
    assert "a at x 100, y 200, z 0 um" in summary and "200 um in XY" in summary
    assert "dim (laser_power 5)" in summary and "z_planes 3" in summary
    assert question["status"] == "needs_go_ahead" and "2 acquisitions" in question["not_done_yet"]
    assert saved_images(microscope) == [] and position(microscope)["x"] == 0.0  # nothing yet

    assert conversation.send("yes, start") == "Saved."
    run = tool_results(conversation)[-1]
    assert run["acquisitions"] == 2 and run["finished"] == "completed"
    files = saved_images(microscope)
    assert len(files) == 6 and run["files_saved"] == 6  # two stacks of three planes
    assert {p.name for p in files} >= {"a_dim_z000.ome.tif", "a_bright_z002.ome.tif"}
    assert len(microscope.images) == 2 and microscope.images[-1][0].shape == (3, 64, 64)
    assert settings(microscope)["laser_power"] == 20.0  # each channel's settings were applied
    assert position(microscope)["x"] == 100.0
    # the last image was described, so the reply can say what was imaged
    assert run["last_image"]["description"] == "Bright spots."
    assert run["last_image"]["statistics"]["max"] > 0


def test_a_model_that_cannot_see_gets_the_last_image_numbers_only(microscope):
    steps = [("plan_acquisition", PLAN), RUN, "Start?", RUN, "Saved."]
    conversation, _ = talk(microscope, *steps)
    conversation.send("take a stack at a")
    conversation.send("yes")
    last = tool_results(conversation)[-1]["last_image"]
    assert "description" not in last and "cannot see" in last["note"]


def test_a_plan_the_operator_declines_is_not_run(microscope):
    steps = [("plan_acquisition", PLAN), RUN, "Shall I start?", "OK, not now."]
    conversation, _ = talk(microscope, *steps)
    conversation.send("take a stack")
    conversation.send("no")
    assert saved_images(microscope) == []


def test_a_plan_from_earlier_in_the_conversation_is_asked_about_again(microscope):
    steps = [
        ("plan_acquisition", PLAN),
        "Shall I start?",
        "OK, not now.",
        RUN,
        "Shall I start it now?",
    ]
    conversation, _ = talk(microscope, *steps)
    conversation.send("plan a stack at a")
    conversation.send("no, later")
    conversation.send("run it now")  # the plan is two messages old: a new question, no run
    assert tool_results(conversation)[-1]["status"] == "needs_go_ahead"
    assert saved_images(microscope) == []


def test_a_far_away_plan_says_so(microscope):
    far = {**PLAN, "positions": [{"x": 4000, "y": 3000, "z": 400}]}
    conversation, _ = talk(microscope, ("plan_acquisition", far), "Shall I? It is far.")
    conversation.send("image over there")
    summary = tool_results(conversation)[0]["summary"]
    assert "4000 um in XY and 400 um in Z" in summary and "This includes a long move." in summary


def test_the_run_images_exactly_the_planned_positions(microscope):
    here = {**PLAN, "positions": []}  # "here" is fixed when the plan is made
    steps = [
        ("plan_acquisition", here),
        ("move_stage", {"x": 100}),
        "Shall I start?",
        RUN,
        "Done.",
    ]
    conversation, _ = talk(microscope, *steps)
    conversation.send("stack here")
    assert "here at x 0, y 0, z 0 um" in tool_results(conversation)[0]["summary"]
    conversation.send("yes")
    assert position(microscope)["x"] == 0.0  # imaged at the planned x, not where the stage went
    assert {p.name for p in saved_images(microscope)} >= {"here_dim_z000.ome.tif"}


def test_time_points_and_ome_zarr(microscope):
    plan = {
        "name": "lapse",
        "channels": [],  # no channels: the settings as they are now
        "options": {"format": "ome-zarr", "z_planes": 2},
        "time_points": 2,
        "interval_s": 0,
    }
    run = ("run_acquisition", {"plan_id": "lapse-1"})
    conversation, _ = talk(microscope, ("plan_acquisition", plan), "Start?", run, "Saved.")
    conversation.send("two time points here")
    assert "2 time points" in tool_results(conversation)[0]["summary"]
    conversation.send("yes")
    result = tool_results(conversation)[-1]
    assert result["acquisitions"] == 2 and result["finished"] == "completed"
    names = [p.name for p in saved_images(microscope)]
    assert names == ["here_t000.ome.zarr", "here_t001.ome.zarr"]
    assert microscope.images[-1][0].shape == (2, 64, 64)


@pytest.mark.parametrize(
    ("change", "code", "message"),
    [
        ({"positions": [{"x": 100, "y": 200, "z": 2000}]}, "limit", "outside the range"),
        ({"positions": [{"x": 1, "y": 1, "z": 0, "name": "c"}] * 2}, "invalid", "more than once"),
        ({"channels": [{"name": "c", "settings": {"power": 5}}]}, "invalid", "'power'"),
        ({"options": {"planes": 3}}, "invalid", "'planes'"),
        ({"options": {"format": "png"}}, "invalid", "'png'"),
    ],
    ids=["outside", "repeated-name", "unknown-setting", "unknown-option", "bad-option-value"],
)
def test_a_plan_with_a_problem_is_refused_before_anything_moves(microscope, change, code, message):
    conversation, _ = talk(microscope, ("plan_acquisition", {**PLAN, **change}), "It cannot run.")
    conversation.send("plan it")
    error = tool_results(conversation)[0]["error"]
    assert error["code"] == code and message in error["message"]
    assert microscope.warnings and position(microscope)["x"] == 0.0
    if "configured_options" in error:
        assert error["advice"] == OPTIONS_ADVICE


def test_an_unknown_setting_in_a_plan_lists_the_drivers_names(microscope):
    bad = {**PLAN, "channels": [{"name": "c", "settings": {"power": 5}}]}
    conversation, _ = talk(microscope, ("plan_acquisition", bad), "Which setting?")
    conversation.send("plan it")
    error = tool_results(conversation)[0]["error"]
    assert error["configured_options"] == ["laser_power", "gain", "exposure_ms", "objective"]


def test_a_refusal_during_a_run_stops_it_and_says_what_was_saved(microscope):
    plan = {**PLAN, "channels": [*PLAN["channels"], {"name": "hot", "settings": {"gain": 900}}]}
    steps = [("plan_acquisition", plan), "Shall I start?", RUN, "The gain was refused."]
    conversation, _ = talk(microscope, *steps)
    conversation.send("a stack at a")
    conversation.send("yes")
    result = tool_results(conversation)[-1]
    assert result["finished"] == "failed" and "outside the limits" in result["error"]["message"]
    assert result["acquisitions"] == 2 and result["files_saved"] == 6
    assert microscope.warnings  # a refusal by the driver shows in the window


def test_stop_ends_a_run_after_the_current_acquisition(microscope):
    plan = {**PLAN, "time_points": 3}
    steps = [("plan_acquisition", plan), "Shall I start?", RUN, "Stopped."]
    conversation, _ = talk(microscope, *steps)
    conversation.send("a stack at a, three times")
    microscope.on_image = lambda image, caption: microscope.stop()  # Stop after the first
    conversation.send("yes")
    result = tool_results(conversation)[-1]
    assert result["finished"] == "stopped" and result["acquisitions"] == 1


def test_a_malformed_plan_goes_back_to_the_model(microscope):
    bad = {**PLAN, "name": "has spaces"}  # a plan name may hold letters, digits, - and _
    conversation, script = talk(
        microscope, ("plan_acquisition", bad), ("plan_acquisition", PLAN), "OK."
    )
    conversation.send("plan it")
    retry = script.requests[1][-1].parts[-1]
    assert retry.part_kind == "retry-prompt"  # Pydantic caught it before our code ran
    assert tool_results(conversation)[0]["plan_id"] == "stack_test-1"


# -- reading the source ----------------------------------------------------------------------


def test_the_source_of_the_agent_the_controller_and_the_driver_can_be_read(microscope):
    steps = [
        ("search_source", {"text": "def set_xyz"}),
        ("read_source", {"file": "zmart_controller/session.py", "start_line": 1, "lines": 3}),
        "Here is how it works.",
    ]
    conversation, _ = talk(microscope, *steps)
    conversation.send("how does a move reach the microscope?")
    found, read = tool_results(conversation)
    assert any(m.startswith("zmart_controller/session.py:") for m in found["matches"])
    assert any(
        m.startswith("mock_zmart_driver/zmart_controller/__init__.py:") for m in found["matches"]
    )
    assert read["lines"].startswith("1 to 3 of") and read["text"].startswith("1: ")


def test_nothing_outside_those_sources_can_be_read(microscope):
    conversation, _ = talk(microscope, ("read_source", {"file": "../../etc/passwd"}), "No.")
    conversation.send("read that file")
    error = tool_results(conversation)[0]["error"]
    assert error["code"] == "not_found" and "zmart_ai_agent/tools.py" in error["configured_options"]
    parts = ("zmart_ai_agent/", "zmart_controller/", "mock_zmart_driver/")
    assert all(f.startswith(parts) for f in error["configured_options"])
    assert microscope.warnings == []  # not a fault at the microscope: no red banner


# -- memory ----------------------------------------------------------------------------------


def answer(n):
    """A model answer with its reasoning attached, as some models send it."""
    return ModelResponse(parts=[ThinkingPart("thinking", signature=f"sig{n}"), TextPart(f"{n}")])


def operator_prompts(history):
    return [
        part.content
        for message in history
        for part in message.parts[:1]
        if isinstance(part, UserPromptPart)
    ]


def test_the_history_only_grows_until_it_is_long(microscope):
    conversation, _ = talk(microscope, *[answer(n) for n in range(1, 16)])
    grown = []
    for n in range(1, 16):
        conversation.send(f"message {n}")
        assert conversation.history[: len(grown)] == grown  # nothing earlier was changed
        grown = list(conversation.history)


def test_a_long_conversation_is_made_smaller_between_turns(microscope):
    steps = [answer(n) for n in range(1, 17)]
    steps[7:7] = [("get_status", {})]  # turn 8 reads the (long) status first
    conversation, _ = talk(microscope, *steps)
    for n in range(1, 17):
        conversation.send(f"message {n}")

    prompts = operator_prompts(conversation.history)
    assert len(prompts) == HISTORY_KEEP_TURNS and prompts[0].startswith("message 7")
    # the newest three turns keep the full state; older ones keep a one-line reading
    assert ["<microscope_state>" in p for p in prompts] == [False] * 7 + [True] * 3
    assert "<microscope_state_then>" in prompts[0] and "position_um" in prompts[0]
    assert "laser_power" in prompts[0] and "serial" not in prompts[0]
    # the old reasoning is left out, all of it, so what remains passes the model's check
    parts = [part for message in conversation.history for part in message.parts]
    assert not any(isinstance(part, ThinkingPart) for part in parts)
    assert any(isinstance(part, TextPart) and part.content == "16" for part in parts)
    (status,) = tool_results(conversation)
    assert status.endswith("(shortened in memory)")


def test_clear_context_forgets_the_conversation(microscope):
    conversation, script = talk(microscope, "One.", "Two.")
    conversation.send("first")
    conversation.clear()
    conversation.send("second")
    assert operator_prompts(script.requests[1]) == [script.requests[1][-1].parts[0].content]
