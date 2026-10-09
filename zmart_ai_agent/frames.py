"""The frames the agent has seen this session, and what the code reads from them.

Every image a look or a run delivers is kept here as a numbered frame: a
small copy of the picture, the time, where it was taken (position and
settings), an optional label, and the measured numbers. The measures that
matter for driving are where the bright signal sits and how far that is from
the centre of the image, in pixels and, when the driver reports its pixel
size, in micrometres, together with the stage move that would bring the
signal to the centre. That move follows from the one rule every ZMART driver
keeps: a saved image and the stage share one frame, in which right is +x and
down is +y. So no calibration of directions is needed; a signal 50 um to the
right of the centre is centred by moving the stage 50 um in +x.

A driver whose camera registration has not been measured yet saves its
pictures turned or mirrored, whatever the rule says. So the ``calibrate``
tool (``looking.py``) can measure how the picture really moves when the
stage moves, and the answer is kept here per microscope and objective
(``Calibration``); from then on the centring move is measured rather than
nominal. How far the picture moved between two frames is measured by phase
correlation of their copies: a standard way of finding the shift between two
pictures of the same scene. The map (``sample_map``) is a summary derived from
the frames, for the microscope state: where the frames put the sample, the
best focus from the frames taken at the same place, and the labelled frames.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-09
License: MIT
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from zmart_controller.registry import config_root

from .schedules import hms
from .settings import (
    CALIBRATION_FILE,
    FRAME_HISTORY_BYTES,
    LOOK_FRAMES_MAX,
    MAP_PLACE_FRAMES,
    MAP_SAME_PLACE_UM,
    MAP_SATURATED_MAX,
    MAP_SIGNAL_MIN,
)

AXES = ("x", "y", "z")
MICROMETRE_WORDS = ("um", "µm", "μm", "micrometer", "micrometre", "micron", "microns")


def pixel_size_um(observed: dict[str, Any] | None) -> tuple[float, float] | None:
    """The size of one pixel in micrometres (across, down), from the driver's read-only
    report, or None when the driver does not say.

    The controller's contract shows it as ``pixel_size: {"x": ..., "y": ...,
    "unit": "um"}``; a plain number is taken as both.
    """
    value = (observed or {}).get("pixel_size")
    if isinstance(value, dict):
        unit = str(value.get("unit", "um")).lower()
        if unit not in MICROMETRE_WORDS:
            return None
        try:
            across, down = float(value.get("x")), float(value.get("y", value.get("x")))
        except (TypeError, ValueError):
            return None
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        across = down = float(value)
    else:
        return None
    return (across, down) if across > 0 and down > 0 else None


def field_um(observed: dict[str, Any] | None, shape: list[int]) -> tuple[float, float] | None:
    """The width and height of an image in micrometres: its pixels times their size."""
    pixel = pixel_size_um(observed)
    if pixel is None or len(shape) < 2:
        return None
    return float(shape[1]) * pixel[0], float(shape[0]) * pixel[1]


def nominal_scale(field: tuple[float, float] | None) -> list[list[float]] | None:
    """How a stage move shows in the picture under the ZMART frame rule: for a move
    of one micrometre on x, then on y, the fraction of the picture's width the
    content moves right and of its height it moves down. Moving the stage +x
    carries the field of view to +x, so the content moves left. None without
    the field of view in micrometres."""
    if field is None:
        return None
    return [[-1.0 / field[0], 0.0], [0.0, -1.0 / field[1]]]


def centre_move(offset: tuple[float, float], scale: list[list[float]]) -> dict[str, float] | None:
    """The x and y move (um) that brings a signal at ``offset`` (fractions of the
    picture right of and below its centre) to the centre, under ``scale`` (see
    ``nominal_scale``): the move whose shift cancels the offset."""
    matrix = np.array(scale, dtype=float).T  # columns: the picture shift per um of x and of y
    try:
        move = np.linalg.solve(matrix, -np.array(offset, dtype=float))
    except np.linalg.LinAlgError:
        return None
    return {"x": round(float(move[0]), 1), "y": round(float(move[1]), 1)}


def direction(right: float, down: float) -> str:
    """Which way the content moves in the picture, in a word, for a move of +1 um."""
    if abs(right) >= abs(down):
        return "right" if right > 0 else "left"
    return "down" if down > 0 else "up"


class Calibration:
    """The measured scale (see ``nominal_scale``) per microscope and objective, kept
    in a JSON file under the computer's ZMART configuration folder, so it
    survives the session. ``path`` is for the tests; by default the file named
    in the settings."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else config_root().joinpath(*CALIBRATION_FILE)
        self._lock = threading.Lock()

    def load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def scale(self, key: str) -> list[list[float]] | None:
        entry = self.load().get(key)
        return entry["scale"] if entry else None

    def store(self, key: str, scale: list[list[float]], when: str, step_um: float) -> None:
        with self._lock:
            data = self.load()
            data[key] = {"scale": scale, "measured": when, "step_um": step_um}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def calibration_key(name: str, where: dict[str, Any] | None) -> str:
    """Which calibration a frame uses: the microscope's name and its objective, as the
    read-only report or the settings name it, else the microscope alone."""
    where = where or {}
    objective = (where.get("observed") or {}).get("objective")
    if objective is None:
        objective = (where.get("settings") or {}).get("objective")
    return name if objective is None else f"{name}/{objective}"


def shift(before: np.ndarray, after: np.ndarray) -> dict[str, float] | None:
    """How far the content moved from ``before`` to ``after``, in pixels (right, down),
    by phase correlation; ``confidence`` is the share of the correlation in its
    peak, from 0 to 1. None when the two copies differ in size."""
    a = np.asarray(before, dtype=float)
    b = np.asarray(after, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        return None
    a, b = a - a.mean(), b - b.mean()
    cross = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    correlation = np.fft.ifft2(cross / (np.abs(cross) + 1e-9)).real
    row, col = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    peak = float(correlation[row, col])

    def centred(index: int, size: int) -> int:
        return index - size if index > size // 2 else index

    def refine(values: np.ndarray, index: int) -> float:  # a parabola through the peak
        left, mid, right = values[index - 1], values[index], values[(index + 1) % len(values)]
        bend = left - 2 * mid + right
        return 0.0 if bend == 0 else 0.5 * (left - right) / bend

    down = centred(row, a.shape[0]) + refine(correlation[:, col], row)
    across = centred(col, a.shape[1]) + refine(correlation[row, :], col)
    return {
        "right": round(float(across), 2),
        "down": round(float(down), 2),
        "confidence": round(peak, 3),
    }


class FrameHistory:
    """The frames of the session, newest last, with their copies capped at
    ``max_bytes`` together; the numbers keep counting when the oldest go.
    Thread-safe: a look runs on its own thread while the window reads the listing."""

    def __init__(
        self,
        clock: Callable[[], float] = time.time,
        max_bytes: int = FRAME_HISTORY_BYTES,
        calibration: Calibration | None = None,
        name: str = "",
    ):
        self.clock = clock
        self.max_bytes = max_bytes
        self.calibration = calibration  # the measured scales, when calibrate may be used
        self.name = name  # the microscope's name, which keys its calibration
        self.frames: list[dict[str, Any]] = []
        self._count = 0
        self._lock = threading.Lock()

    def add(
        self,
        copy: np.ndarray,
        stats: dict[str, Any],
        source: str,
        where: dict[str, Any] | None,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Keep a frame. ``copy`` is its small copy (``images.small_copy``), ``stats`` the
        numbers measured on the full image (``images.image_statistics``), ``source``
        what delivered it ("look", or a run's label), ``where`` the position, settings
        and observed report it was taken in. Returns the entry."""
        where = where or {}
        position = where.get("position_um") or {}
        field = field_um(where.get("observed"), stats.get("shape") or list(copy.shape))
        key = calibration_key(self.name, where)
        measured = self.calibration.scale(key) if self.calibration is not None else None
        scale = measured or nominal_scale(field)
        with self._lock:
            self._count += 1
            entry: dict[str, Any] = {
                "n": self._count,
                "t": self.clock(),
                "source": source,
                "position": {axis: position.get(axis) for axis in AXES},
                "settings": dict(where.get("settings") or {}),
                "objective": (where.get("observed") or {}).get("objective"),
                "calibration": key,
                "measures": _measures(stats, field, scale, "measured" if measured else "nominal"),
                "image": np.asarray(copy),
                "field_um": field,
            }
            if label:
                entry["label"] = str(label)
            self.frames.append(entry)
            while len(self.frames) > 1 and sum(_bytes(f) for f in self.frames) > self.max_bytes:
                self.frames.pop(0)
            return entry

    def pick(self, chosen: Any) -> list[dict[str, Any]]:
        """The frames ``chosen`` names: "last 3", "1,7", "3-10", a count or a list of
        numbers; at most LOOK_FRAMES_MAX. A ValueError says what is wrong, naming
        the numbers the history holds."""
        with self._lock:
            frames = list(self.frames)
        by_number = {f["n"]: f for f in frames}
        held = f"{frames[0]['n']} to {frames[-1]['n']}" if frames else "none"
        chosen = _wanted(chosen)
        if isinstance(chosen, int):
            picked = frames[-chosen:] if chosen > 0 else []
        elif isinstance(chosen, tuple):
            low, high = chosen
            picked = [f for f in frames if low <= f["n"] <= high]
        else:
            missing = [n for n in chosen if n not in by_number]
            if missing:
                raise ValueError(
                    f"no frame {', '.join(map(str, missing))}; the history holds {held}"
                )
            picked = [by_number[n] for n in chosen]
        if len(picked) > LOOK_FRAMES_MAX:
            raise ValueError(f"at most {LOOK_FRAMES_MAX} frames in one look")
        return picked

    def brief(self, entry: dict[str, Any]) -> dict[str, Any]:
        """A frame for the model: its number, time, source, label, position, settings
        and measures."""
        out: dict[str, Any] = {"n": entry["n"], "time": hms(entry["t"]), "source": entry["source"]}
        if entry.get("label"):
            out["label"] = entry["label"]
        out["position_um"] = entry["position"]
        out["settings"] = entry["settings"]
        out.update(entry["measures"])
        return out

    def placing(self, entry: dict[str, Any]) -> dict[str, Any]:
        """The new frame in a look's answer: its number, time and label, and where the
        signal sits with the move that would centre it. The raw numbers come
        with the look's statistics, so they are not repeated here."""
        keys = ("offset_px", "offset_um", "centre_move_um", "scale", "signal")
        out: dict[str, Any] = {"n": entry["n"], "time": hms(entry["t"])}
        if entry.get("label"):
            out["label"] = entry["label"]
        out.update({k: v for k, v in entry["measures"].items() if k in keys})
        return out

    def compare(self, picked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """For each frame after the first: how it differs from the first frame shown and
        from the one before it. The picture's shift (in pixels, and in um when
        the pixel size is known), by phase correlation when the two were taken with
        the same objective, and the change in sharpness and peak."""
        out = []
        for index, entry in enumerate(picked[1:], 1):
            row: dict[str, Any] = {"n": entry["n"]}
            for name, other in (("since_first", picked[0]), ("since_previous", picked[index - 1])):
                row[name] = _change(other, entry)
            out.append(row)
        return out

    def listing(self) -> dict[str, Any] | None:
        """For the microscope state: how many frames, the labels, and the last three in brief."""
        with self._lock:
            frames = list(self.frames)
        if not frames:
            return None
        labels = {f["label"]: f["n"] for f in frames if f.get("label")}
        last = [
            {
                "n": f["n"],
                "time": hms(f["t"]),
                "source": f["source"],
                **{
                    k: f["measures"].get(k)
                    for k in ("peak", "saturated_percent", "sharpness", "centre_move_um")
                },
            }
            for f in frames[-3:]
        ]
        return {
            "count": len(frames),
            "numbers": f"{frames[0]['n']}-{frames[-1]['n']}",
            "labels": labels,
            "last": last,
        }

    def clear(self) -> None:
        """Forget every frame; Clear context does this."""
        with self._lock:
            self.frames, self._count = [], 0


def _wanted(chosen: Any) -> int | tuple[int, int] | list[int]:
    """``frames`` as the model gives it, in one of three shapes: a count (the last
    so many), a range, or a list of frame numbers."""
    if isinstance(chosen, bool):
        raise ValueError("frames is 'last 3', a list of frame numbers, or a range like '3-10'")
    if isinstance(chosen, int):
        return chosen
    if isinstance(chosen, str):
        text = chosen.strip().lower()
        if last := re.fullmatch(r"(?:last\s*)?(\d+)", text):
            return int(last.group(1))
        if span := re.fullmatch(r"(\d+)\s*-\s*(\d+)", text):
            return int(span.group(1)), int(span.group(2))
        if re.fullmatch(r"\d+(\s*,\s*\d+)+", text):
            return [int(n) for n in text.split(",")]
    if isinstance(chosen, list) and all(
        isinstance(n, int) and not isinstance(n, bool) for n in chosen
    ):
        return list(chosen)
    raise ValueError("frames is 'last 3', a list of frame numbers, or a range like '3-10'")


def _bytes(entry: dict[str, Any]) -> int:
    image = entry.get("image")
    return image.nbytes if image is not None else 0


def _measures(
    stats: dict[str, Any],
    field: tuple[float, float] | None,
    scale: list[list[float]] | None,
    scale_kind: str,
) -> dict[str, Any]:
    """What the code reads from one image's numbers: how bright and how sharp it is,
    and where its signal sits.

    ``peak``, ``contrast`` and ``mean`` are fractions of the camera's full
    range. ``offset_px`` is how far the signal's centre lies right of and below
    the image centre, in pixels; ``offset_um`` the same in micrometres when the
    driver reports its pixel size; and ``centre_move_um`` the stage move, in x
    and y, that would bring the signal to the centre, under ``scale``: the
    measured one when calibrate has run for this objective, else the nominal
    one from the ZMART frame rule (right is +x, down is +y); ``scale`` says
    which.
    """
    full = float(stats.get("full_scale") or 1.0)
    out: dict[str, Any] = {
        "peak": round(float(stats.get("max", 0.0)) / full, 3),
        "contrast": round(
            (float(stats.get("max", 0.0)) - float(stats.get("background", 0.0))) / full, 4
        ),
        "mean": round(float(stats.get("mean", 0.0)) / full, 4),
        "saturated_percent": float(stats.get("saturated_percent", 0.0)),
        "sharpness": float(stats.get("sharpness", 0.0)),
    }
    centroid = stats.get("signal_centroid")
    if centroid is None:
        out["signal"] = "none"
        return out
    shape = stats.get("shape") or [0, 0]
    right, down = centroid["col"] - 0.5, centroid["row"] - 0.5  # fractions of the image
    out["offset_px"] = {"right": round(right * shape[1], 1), "down": round(down * shape[0], 1)}
    if field is not None:
        out["offset_um"] = {"right": round(right * field[0], 1), "down": round(down * field[1], 1)}
    if scale is not None and (move := centre_move((right, down), scale)) is not None:
        out["centre_move_um"] = move
        out["scale"] = scale_kind
    return out


def _change(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    change: dict[str, Any] = {
        "seconds": round(after["t"] - before["t"], 1),
        "sharpness": round(after["measures"]["sharpness"] - before["measures"]["sharpness"], 4),
        "peak": round(after["measures"]["peak"] - before["measures"]["peak"], 3),
    }
    a, b, field = before.get("image"), after.get("image"), after.get("field_um")
    if (
        a is not None
        and b is not None
        and a.shape == b.shape
        and before.get("objective") == after.get("objective")
    ):
        moved = shift(a, b)
        if moved is not None:
            change["image_shift_px"] = {"right": moved["right"], "down": moved["down"]}
            if field:
                change["image_shift_um"] = {
                    "right": round(moved["right"] * field[0] / b.shape[1], 1),
                    "down": round(moved["down"] * field[1] / b.shape[0], 1),
                }
            change["shift_confidence"] = moved["confidence"]
    return change


def flag(entry: dict[str, Any]) -> str | None:
    """Why a frame cannot be measured from ("no signal", "saturated"), or None."""
    measures = entry["measures"]
    if measures.get("signal") == "none" or measures["contrast"] < MAP_SIGNAL_MIN:
        return "no signal"
    if measures["saturated_percent"] > MAP_SATURATED_MAX:
        return "saturated"
    return None


def sample_map(history: FrameHistory) -> dict[str, Any] | None:
    """The map, for the microscope state: where the stage would centre the sample,
    the best focus from the frames taken at the newest frame's place, and the
    labelled frames. Frames that cannot be measured from are left out and
    named. None while there are no frames."""
    with history._lock:
        frames = list(history.frames)
    if not frames:
        return None
    now = history.clock()
    usable = [f for f in frames if flag(f) is None]
    out: dict[str, Any] = {}
    if (place := _place(usable, now)) is not None:
        out["sample_at"] = place
    if (focus := _focus(usable, now)) is not None:
        out["best_focus"] = focus
    labels = {
        f["label"]: {
            **{axis: f["position"][axis] for axis in AXES},
            "frame": f["n"],
            "age_s": int(now - f["t"]),
        }
        for f in frames
        if f.get("label")
    }
    if labels:
        out["labels"] = labels
    left_out = {f["n"]: flag(f) for f in frames if flag(f)}
    if left_out:
        out["left_out"] = left_out
    return out


def _place(usable: list[dict[str, Any]], now: float) -> dict[str, Any] | None:
    """Where the stage would centre the sample: each frame's position plus its centring
    move, the median over the last few usable frames."""
    found = [
        (
            f["position"]["x"] + f["measures"]["centre_move_um"]["x"],
            f["position"]["y"] + f["measures"]["centre_move_um"]["y"],
            f,
        )
        for f in usable[-MAP_PLACE_FRAMES:]
        if f["measures"].get("centre_move_um") and f["position"]["x"] is not None
    ]
    if not found:
        return None
    return {
        "x": round(float(np.median([p[0] for p in found])), 1),
        "y": round(float(np.median([p[1] for p in found])), 1),
        "frames": [p[2]["n"] for p in found],
        "age_s": int(now - found[-1][2]["t"]),
    }


def _focus(usable: list[dict[str, Any]], now: float) -> dict[str, Any] | None:
    """The best focus from the frames at the newest frame's place (within
    MAP_SAME_PLACE_UM in x and y): the top of a parabola through the sharpest
    frame and its neighbours in z, with half their spacing as the uncertainty.
    At the edge of the frames' range, the edge, and which way to search."""
    if not usable or usable[-1]["position"]["z"] is None:
        return None
    here = usable[-1]["position"]
    curve: dict[float, float] = {}
    for f in usable:
        p = f["position"]
        if all(p[a] is not None and abs(p[a] - here[a]) <= MAP_SAME_PLACE_UM for a in ("x", "y")):
            curve[p["z"]] = max(curve.get(p["z"], 0.0), f["measures"]["sharpness"])
    if len(curve) < 2:
        return None
    positions = sorted(curve)
    index = max(range(len(positions)), key=lambda i: curve[positions[i]])
    best = positions[index]
    out: dict[str, Any] = {
        "z": round(best, 1),
        "frames_used": len(positions),
        "age_s": int(now - usable[-1]["t"]),
    }
    if index in (0, len(positions) - 1):
        out["edge"] = "search lower z" if index == 0 else "search higher z"
        spacing = positions[1] - positions[0] if index == 0 else positions[-1] - positions[-2]
        out["plus_minus"] = round(abs(spacing), 1)
        return out
    (z0, z1, z2), (v0, v1, v2) = (
        positions[index - 1 : index + 2],
        [curve[p] for p in positions[index - 1 : index + 2]],
    )
    denominator = (z0 - z1) * (z0 - z2) * (z1 - z2)
    a = (z2 * (v1 - v0) + z1 * (v0 - v2) + z0 * (v2 - v1)) / denominator if denominator else 0.0
    b = (
        (z2**2 * (v0 - v1) + z1**2 * (v2 - v0) + z0**2 * (v1 - v2)) / denominator
        if denominator
        else 0.0
    )
    if a < 0:
        out["z"] = round(float(np.clip(-b / (2 * a), z0, z2)), 1)
    out["plus_minus"] = round(min(z1 - z0, z2 - z1) / 2, 1)
    return out
