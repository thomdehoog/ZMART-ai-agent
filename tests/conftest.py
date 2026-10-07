"""The mock microscope from ZMART-controller, plugged into the real controller.

The tests drive the pretend microscope that ships with the controller
(see ``mock_microscope.py``), through the controller itself, exactly as the
agent drives a real microscope.

Each test gets its own configuration folder and image folder, so no test
ever writes to the computer's real ZMART configuration.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import pytest
from mock_microscope import mock_connection


@pytest.fixture(autouse=True)
def _config_in_a_temporary_folder(tmp_path, monkeypatch):
    """Never let a test read or write the real ZMART configuration folder."""
    monkeypatch.setenv("ZMART_MICROSCOPY_ROOT", str(tmp_path / "config"))


@pytest.fixture
def connection(tmp_path) -> dict:
    """What the mock is told when it connects: quick, saving in this test's own folder."""
    return mock_connection(tmp_path / "images")
