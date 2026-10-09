"""The frames: the measures read from a picture, the frame history, the map, and calibrate.

The pure pieces are checked on pictures drawn here; the whole, from a look to
the map, on the mock microscope through the real controller.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-09
License: MIT
"""

import json

import numpy as np
import pytest
from test_agent import position, talk, tool_results

from zmart_ai_agent.frames import (
    Calibration,
    FrameHistory,
    centre_move,
    field_um,
    nominal_scale,
    pixel_size_um,
    sample_map,
    shift,
)
from zmart_ai_agent.images import image_statistics, sharpness, small_copy
from zmart_ai_agent.settings import LOOK_FRAMES_MAX


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def disc(row: int, col: int, radius: int = 6, shape=(64, 96), value: int = 4000) -> np.ndarray:
    """A camera picture with one bright disc at (row, col), on a dark level of 100."""
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    picture = np.full(shape, 100, dtype=np.uint16)
    picture[(rows - row) ** 2 + (cols - col) ** 2 <= radius**2] = value
    return picture


OBSERVED = {"objective": "10x", "pixel_size": {"x": 2.0, "y": 2.0, "unit": "um"}}
WHERE = {"position_um": {"x": 100.0, "y": 50.0, "z": 0.0}, "settings": {}, "observed": OBSERVED}


# -- the numbers read from a picture ---------------------------------------------------------


def test_the_statistics_say_where_the_signal_sits():
    stats = image_statistics(disc(32, 72))  # right of the centre (col 48), on the middle row
    assert stats["shape"] == [64, 96] and stats["full_scale"] == 65535
    assert stats["background"] == 100 and stats["saturated_percent"] == 0
    assert abs(stats["signal_centroid"]["col"] - 72 / 96) < 0.01
    assert abs(stats["signal_centroid"]["row"] - 32 / 64) < 0.01
    assert image_statistics(np.full((64, 96), 100, dtype=np.uint16))["signal_centroid"] is None


def test_sharpness_peaks_in_focus_and_ignores_the_brightness():
    sharp = disc(32, 48).astype(float)
    soft = sharp.copy()
    for _ in range(4):  # a repeated box blur looks like defocus
        padded = np.pad(soft, 2, mode="edge")
        soft = sum(padded[i : i + 64, j : j + 96] for i in range(5) for j in range(5)) / 25
    assert sharpness(sharp) > sharpness(soft) > 0
    assert abs(sharpness(sharp) - sharpness(sharp * 3)) < 1e-3  # the same at any exposure


def test_the_pixel_size_and_the_field_come_from_the_observed_report():
    assert pixel_size_um(OBSERVED) == (2.0, 2.0)
    assert pixel_size_um({"pixel_size": 0.5}) == (0.5, 0.5)
    assert pixel_size_um({"pixel_size": {"x": 1, "y": 1, "unit": "mm"}}) is None
    assert pixel_size_um({}) is None and pixel_size_um(None) is None
    assert field_um(OBSERVED, [64, 96]) == (192.0, 128.0)
    assert field_um({}, [64, 96]) is None


def test_the_nominal_centring_move_follows_the_frame_rule():
    # Under the rule (right is +x, down is +y), a signal 20 um right of and 10 um
    # below the centre is centred by moving the stage +20 in x and +10 in y.
    scale = nominal_scale((192.0, 128.0))
    assert centre_move((20 / 192, 10 / 128), scale) == {"x": 20.0, "y": 10.0}
    assert nominal_scale(None) is None
    assert centre_move((0.1, 0.1), [[0.0, 0.0], [0.0, 0.0]]) is None  # nothing to solve


def test_the_shift_between_two_pictures_is_measured():
    picture = np.random.default_rng(1).random((64, 96))
    moved = shift(picture, np.roll(np.roll(picture, 3, axis=1), -2, axis=0))
    assert (moved["right"], moved["down"]) == (3.0, -2.0) and moved["confidence"] > 0.9
    assert shift(picture, picture[:32]) is None  # sizes differ


# -- the history -----------------------------------------------------------------------------


def kept(history: FrameHistory, picture: np.ndarray, where=WHERE, **more):
    return history.add(small_copy(picture), image_statistics(picture), "look", where, **more)


def test_a_frame_is_kept_with_its_measures_and_the_move_that_centres_it():
    history = FrameHistory(Clock())
    entry = kept(history, disc(32, 72), label="before")
    measures = entry["measures"]
    assert entry["n"] == 1 and entry["label"] == "before"
    assert entry["position"] == {"x": 100.0, "y": 50.0, "z": 0.0}
    assert measures["offset_px"] == {"right": 24.0, "down": 0.0}
    assert measures["offset_um"] == {"right": 48.0, "down": 0.0}  # 24 px of 2 um
    assert measures["centre_move_um"] == {"x": 48.0, "y": 0.0} and measures["scale"] == "nominal"
    assert 0 < measures["peak"] < 0.1 and measures["saturated_percent"] == 0
    brief = history.brief(entry)
    assert brief["time"] and brief["source"] == "look" and brief["centre_move_um"]["x"] == 48.0
    placing = history.placing(entry)
    assert set(placing) == {
        "n",
        "time",
        "label",
        "offset_px",
        "offset_um",
        "centre_move_um",
        "scale",
    }


def test_without_a_pixel_size_the_offset_is_in_pixels_only():
    history = FrameHistory(Clock())
    entry = kept(history, disc(32, 72), where={"position_um": {"x": 0, "y": 0, "z": 0}})
    measures = entry["measures"]
    assert measures["offset_px"] == {"right": 24.0, "down": 0.0}
    assert "offset_um" not in measures and "centre_move_um" not in measures


def test_frames_are_picked_by_count_list_or_range():
    history = FrameHistory(Clock())
    for col in (20, 30, 40, 50):
        kept(history, disc(32, col))
    assert [f["n"] for f in history.pick("last 2")] == [3, 4]
    assert [f["n"] for f in history.pick("1,4")] == [1, 4]
    assert [f["n"] for f in history.pick("2-3")] == [2, 3]
    assert [f["n"] for f in history.pick([4, 1])] == [4, 1]
    assert [f["n"] for f in history.pick(3)] == [2, 3, 4]
    with pytest.raises(ValueError, match="no frame 9; the history holds 1 to 4"):
        history.pick("1,9")
    with pytest.raises(ValueError, match="frames is 'last 3'"):
        history.pick("yesterday")
    for col in range(LOOK_FRAMES_MAX):
        kept(history, disc(32, 48))
    with pytest.raises(ValueError, match=f"at most {LOOK_FRAMES_MAX}"):
        history.pick(f"1-{LOOK_FRAMES_MAX + 1}")


def test_the_oldest_copies_go_but_the_numbers_keep_counting():
    one = small_copy(disc(32, 48)).nbytes
    history = FrameHistory(Clock(), max_bytes=2 * one)
    for col in (20, 30, 40):
        kept(history, disc(32, col))
    assert [f["n"] for f in history.frames] == [2, 3]
    assert history.listing()["numbers"] == "2-3" and history.listing()["count"] == 2
    history.clear()
    assert history.frames == [] and history.listing() is None


def test_comparing_frames_measures_the_shift_and_the_change():
    clock = Clock()
    history = FrameHistory(clock)
    first = kept(history, disc(32, 40))
    clock.now += 30
    second = kept(history, disc(32, 46))  # the disc moved 6 px right: 12 um
    (change,) = history.compare([first, second])
    assert change["n"] == 2 and change["since_first"]["seconds"] == 30
    assert change["since_first"]["image_shift_px"]["right"] == pytest.approx(6, abs=0.5)
    assert change["since_first"]["image_shift_um"]["right"] == pytest.approx(12, abs=1)
    assert change["since_first"]["shift_confidence"] > 0.1
    assert change["since_first"] == change["since_previous"]


def test_the_map_says_where_the_sample_is_and_the_best_focus():
    clock = Clock()
    history = FrameHistory(clock)
    # Three frames at the same place, 2 um apart in z, the middle one sharpest
    # (blurred less), plus one with no signal that is left out.
    for z, blur in ((-2, 2), (0, 0), (2, 2)):
        picture = disc(32, 60).astype(float)
        for _ in range(blur):
            padded = np.pad(picture, 2, mode="edge")
            picture = sum(padded[i : i + 64, j : j + 96] for i in range(5) for j in range(5)) / 25
        where = {**WHERE, "position_um": {"x": 100.0, "y": 50.0, "z": float(z)}}
        kept(history, picture.astype(np.uint16), where=where)
    kept(history, np.full((64, 96), 100, dtype=np.uint16), label="dark")
    clock.now += 10
    found = sample_map(history)
    assert found["sample_at"]["x"] == 124.0 and found["sample_at"]["y"] == 50.0  # 12 px right
    assert found["sample_at"]["frames"] == [1, 2, 3] and found["sample_at"]["age_s"] == 10
    assert found["best_focus"]["z"] == 0.0 and found["best_focus"]["frames_used"] == 3
    assert found["best_focus"]["plus_minus"] == 1.0 and "edge" not in found["best_focus"]
    assert found["labels"]["dark"]["frame"] == 4 and found["left_out"] == {4: "no signal"}


def test_a_calibration_is_kept_per_microscope_and_objective(tmp_path):
    calibration = Calibration(tmp_path / "calibration.json")
    assert calibration.scale("mock/10x") is None
    calibration.store("mock/10x", [[0.0, -0.01], [-0.01, 0.0]], "12:00:00", 20.0)
    assert Calibration(tmp_path / "calibration.json").scale("mock/10x") == [
        [0.0, -0.01],
        [-0.01, 0.0],
    ]
    assert json.loads((tmp_path / "calibration.json").read_text())["mock/10x"]["step_um"] == 20.0
    history = FrameHistory(Clock(), calibration=calibration, name="mock")
    # 24 px right of the centre: with this camera, +y carries the content left, so
    # a +y move of a quarter of the field (0.25 / 0.01 per um) centres it.
    entry = kept(history, disc(32, 72))
    assert entry["calibration"] == "mock/10x" and entry["measures"]["scale"] == "measured"
    assert entry["measures"]["centre_move_um"] == {"x": 0.0, "y": 25.0}


# -- on the mock microscope, through the controller ------------------------------------------


def test_a_look_keeps_a_frame_and_the_state_carries_the_frames_and_the_map(microscope):
    look = ("look", {"question": "where is it?", "label": "start"})
    conversation, script = talk(microscope, look, "Here.", "Hello.")
    conversation.send("look")
    frame = tool_results(conversation)[0]["frame"]
    assert frame["n"] == 1 and frame["label"] == "start" and frame["scale"] == "nominal"
    assert "centre_move_um" in frame and "offset_um" in frame  # the mock reports its pixel size
    conversation.send("hi")
    content = script.requests[-1][-1].parts[-1].content
    state = json.loads(content.split("<microscope_state>")[1][: -len("</microscope_state>")])
    assert state["frames"]["count"] == 1 and state["frames"]["labels"] == {"start": 1}
    assert state["frames"]["last"][0]["n"] == 1 and "sharpness" in state["frames"]["last"][0]
    assert "sample_at" in state["map"] and state["map"]["labels"]["start"]["frame"] == 1


def test_calibrate_measures_how_the_picture_moves_and_is_agreed_first(microscope):
    # The mock's camera is turned a quarter turn, and its shipped registration does
    # not know that yet: the pictures do not follow the frame rule until the
    # microscope is set up. calibrate finds that out and keeps it.
    steps = [("calibrate", {}), "Shall I?", ("calibrate", {}), "Measured."]
    conversation, _ = talk(microscope, *steps)
    conversation.send("calibrate the camera")
    question = tool_results(conversation)[0]
    assert question["status"] == "needs_go_ahead" and "in x and back" in question["not_done_yet"]
    assert len(microscope.frames.frames) == 0
    conversation.send("yes")
    report = tool_results(conversation)[-1]
    assert report["frames"] == [1, 2, 3] and report["follows_the_frame_rule"] is False
    assert report["x"]["content_moves"] == "up" and report["y"]["content_moves"] == "left"
    assert report["kept_as"] == "mock_microscope/10x/0.30 Air"
    assert position(microscope) == {"x": 0.0, "y": 0.0, "z": 0.0}  # back where it started
    assert microscope.frames.calibration.scale(report["kept_as"]) is not None
    # from now on the centring move is measured, and it goes the way the camera is turned
    conversation, _ = talk(microscope, ("look", {"question": "where?"}), "There.")
    conversation.send("look")
    frame = tool_results(conversation)[0]["frame"]
    assert frame["scale"] == "measured"
    move, offset = frame["centre_move_um"], frame["offset_um"]
    assert move["x"] == pytest.approx(offset["down"], abs=0.6)  # a +x move carries it up
    assert move["y"] == pytest.approx(offset["right"], abs=0.6)  # a +y move carries it left


def test_calibrate_refuses_a_picture_without_a_sample(microscope, monkeypatch):
    import tifffile
    from mock_microscope import MOCK_OPS

    original = MOCK_OPS["acquire"]

    def dark(handle, **kwargs):
        answer = original(handle, **kwargs)
        for path in answer["content"]["files"]:
            if path.endswith(".tif"):
                tifffile.imwrite(path, np.full((64, 64), 100, dtype=np.uint16))
        return answer

    monkeypatch.setitem(MOCK_OPS, "acquire", dark)
    steps = [("calibrate", {}), "Shall I?", ("calibrate", {}), "Nothing to see."]
    conversation, _ = talk(microscope, *steps)
    conversation.send("calibrate")
    conversation.send("yes")
    error = tool_results(conversation)[-1]["error"]
    assert error["code"] == "invalid" and "no signal" in error["message"]
    assert position(microscope) == {"x": 0.0, "y": 0.0, "z": 0.0}
