"""The mock microscope from ZMART-controller, plugged into the real controller.

The tests drive the pretend microscope that ships with the controller
(see ``mock_microscope.py``), through the controller itself, exactly as the
agent drives a real microscope.

Each test gets its own configuration folder and image folder, so no test
ever writes to the computer's real ZMART configuration. ``microscope`` is a
connected mock, with the window's side (images, warnings, tool calls)
collected in lists for the test to check.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import pytest
from mock_microscope import DRIVER, mock_connection

from zmart_ai_agent.microscope import Microscope


@pytest.fixture(autouse=True)
def _config_in_a_temporary_folder(tmp_path, monkeypatch):
    """Never let a test read or write the real ZMART configuration folder."""
    monkeypatch.setenv("ZMART_MICROSCOPY_ROOT", str(tmp_path / "config"))


@pytest.fixture
def connection(tmp_path) -> dict:
    """What the mock is told when it connects: quick, saving in this test's own folder."""
    return mock_connection(tmp_path / "images")


@pytest.fixture
def microscope(connection):
    """A connected mock microscope, without a vision model unless the test gives one."""
    scope = Microscope(DRIVER, connection)
    scope.connect()
    scope.vision = False  # no vision model in most tests; those with one say so
    scope.images, scope.warnings, scope.tools = [], [], []
    scope.on_image = lambda image, caption: scope.images.append((image, caption))
    scope.on_warning = scope.warnings.append
    scope.on_tool = lambda name, args: scope.tools.append((name, args))
    yield scope
    scope.disconnect()
