"""Where the mock microscope is, and how the tests and the evaluation plug it in.

The mock is the pretend microscope that ships with ZMART-controller
(``tests/mock_zmart_driver`` in a clone of it). Its folder is taken from
``ZMART_MOCK_DRIVER``, or from a ZMART-controller clone next to this
repository. It is plugged into the real controller for this session only,
so nothing is written to the computer's list of drivers.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import zmart_controller
from zmart_controller import utils

MOCK_DRIVER = Path(
    os.environ.get("ZMART_MOCK_DRIVER")
    or Path(__file__).resolve().parent.parent.parent
    / "ZMART-controller"
    / "tests"
    / "mock_zmart_driver"
)
MOCK = {"vendor": "mock", "microscope": "mock-scope", "api": "mock-api"}


def plug_in_mock() -> bool:
    """Plug the mock driver in for this session, once; False when it is not there."""
    if not MOCK_DRIVER.is_dir():
        return False
    if tuple(MOCK.values()) not in utils.REGISTRY:
        zmart_controller.register_driver(MOCK_DRIVER, remember=False)
    return True


def mock_instrument(output_root: str | Path) -> dict[str, Any]:
    """The mock as get_instruments lists it, made quick and saving to ``output_root``.

    ``mock_timing`` and ``output_root`` are the mock's own connection entries:
    moves and images finish at once, and images go to the folder given.
    """
    listed = next(
        i
        for i in zmart_controller.get_instruments()
        if all(i[key] == value for key, value in MOCK.items())
    )
    return {**listed, "mock_timing": "instant", "output_root": str(output_root)}


def mock_ops() -> dict[str, Any]:
    """The mock's functions as the controller holds them.

    The controller hands every session this very dictionary, so a test can
    replace one function here (and put it back) to make the driver answer
    differently, without changing the agent or the controller.
    """
    return utils.REGISTRY[tuple(MOCK.values())]["ops"]
