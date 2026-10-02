"""The mock microscope from ZMART-controller, plugged into the real controller.

The tests drive the pretend microscope that ships with the controller
(see ``mock_microscope.py`` for where it is found), through the controller itself,
exactly as the assistant drives a real microscope. Without it, every test
that needs a microscope is skipped and says why.

Each test gets its own configuration folder and image folder, so no test
ever writes to the computer's real ZMART configuration.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import pytest
from mock_microscope import MOCK_DRIVER, mock_instrument, plug_in_mock

MOCK_AVAILABLE = plug_in_mock()


@pytest.fixture(autouse=True)
def _config_in_a_temporary_folder(tmp_path, monkeypatch):
    """Never let a test read or write the real ZMART configuration folder."""
    monkeypatch.setenv("ZMART_MICROSCOPY_ROOT", str(tmp_path / "config"))


@pytest.fixture
def instrument(tmp_path) -> dict:
    """The mock microscope, quick, saving its images in this test's own folder."""
    if not MOCK_AVAILABLE:
        pytest.skip(f"the mock driver is not at {MOCK_DRIVER}; set ZMART_MOCK_DRIVER")
    return mock_instrument(tmp_path / "images")
