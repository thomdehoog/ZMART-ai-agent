"""The mock microscope the tests and the evaluation drive: a driver module of its own.

The pretend microscope ships with the controller as ``zmart_controller.mock``.
This module is a driver too: it holds one function per command, as every
ZMART driver does, and each function hands the call on to the mock. It is
plugged in exactly as a real driver is::

    zmart_controller.set_instrument(mock_microscope, mock_connection(folder))

Why not plug in ``zmart_controller.mock`` itself? The controller copies a
driver's functions when it connects, so replacing one of the mock's
functions afterwards would not reach a session that is already open. Here
every call looks its function up in ``MOCK_OPS`` at the moment it is made,
so a test can replace one entry (and put it back) to make the microscope
answer differently, even in the middle of a conversation.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import zmart_controller.mock
from zmart_controller.utils import OPS

# The mock's own functions, one per command. A test replaces an entry here.
MOCK_OPS: dict[str, Any] = {
    name: getattr(zmart_controller.mock, name) for name in (*OPS, "disconnect")
}


def _hand_on(name: str):
    def command(*args, **kwargs):
        return MOCK_OPS[name](*args, **kwargs)

    command.__name__ = name
    return command


for _name in MOCK_OPS:
    globals()[_name] = _hand_on(_name)

# This module, to pass to set_instrument or to Microscope as the driver.
DRIVER = sys.modules[__name__]


def mock_connection(output_root: str | Path) -> dict[str, Any]:
    """What the mock is told when it connects: be quick, and save to ``output_root``.

    ``mock_timing`` and ``output_root`` are the mock's own connection entries:
    with "instant", moves and images finish at once instead of taking as long
    as on a real microscope.
    """
    return {"mock_timing": "instant", "output_root": str(output_root)}
