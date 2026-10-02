"""A chat assistant that drives any microscope with a ZMART driver, through the ZMART Controller.

    zmart-ai-agent                                  # the window (a command)
    from zmart_ai_agent.agent import Assistant      # the assistant, without the window

It knows nothing about a microscope in advance: it learns each one from its
ZMART driver when it connects.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

__version__ = "0.1.0"
