"""The mock microscope from ZMART-controller, plugged into the real controller.

The tests drive the pretend microscope that ships with the controller
(``tests/mock_zmart_driver`` in a clone of ZMART-controller), through the
controller itself, exactly as the assistant drives a real microscope. The
mock's folder is taken from ``ZMART_MOCK_DRIVER``, or from a ZMART-controller
clone next to this repository. Without it, every test that needs a
microscope is skipped and says why.

Each test gets its own configuration folder and image folder, so no test
ever writes to the computer's real ZMART configuration.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import zmart_controller

HERE = Path(__file__).resolve().parent
MOCK_DRIVER = Path(
    os.environ.get("ZMART_MOCK_DRIVER")
    or HERE.parent.parent / "ZMART-controller" / "tests" / "mock_zmart_driver"
)
MOCK = {"vendor": "mock", "microscope": "mock-scope", "api": "mock-api"}

if MOCK_DRIVER.is_dir():
    zmart_controller.register_driver(MOCK_DRIVER, remember=False)


@pytest.fixture(autouse=True)
def _config_in_a_temporary_folder(tmp_path, monkeypatch):
    """Never let a test read or write the real ZMART configuration folder."""
    monkeypatch.setenv("ZMART_MICROSCOPY_ROOT", str(tmp_path / "config"))


@pytest.fixture
def instrument(tmp_path) -> dict:
    """The mock microscope as get_instruments lists it, made quick and given its own folder.

    ``mock_timing`` and ``output_root`` are the mock's own connection entries:
    moves and images finish at once, and images go to this test's folder.
    """
    if not MOCK_DRIVER.is_dir():
        pytest.skip(f"the mock driver is not at {MOCK_DRIVER}; set ZMART_MOCK_DRIVER")
    listed = next(
        i
        for i in zmart_controller.get_instruments()
        if all(i[key] == value for key, value in MOCK.items())
    )
    return {**listed, "mock_timing": "instant", "output_root": str(tmp_path / "images")}
