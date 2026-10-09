"""Every constant of the agent, in one place.

The model's instructions and the advice it is given with a refusal are prose
and stay in ``instructions.py``; the numbers and names that one might want to
change are all here. Times are seconds, distances micrometres.

Nothing here describes a particular microscope. What a microscope can do,
its travel range, its settings and its routines, comes from its ZMART driver
when the agent connects.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from typing import Any

# -- the model -------------------------------------------------------------------------
# The providers the window offers. vision: the model can be shown a camera image, so
# the "look" tool gets a real answer about the picture; without it the tool answers
# from the measured numbers only.
PROVIDERS: dict[str, dict[str, Any]] = {
    "Gemini": {
        "kind": "google",
        "model": "gemini-3.5-flash-lite",  # generous free tier, native tool calling
        "key_env": "GEMINI_API_KEY",
        "key_env_also": "GOOGLE_API_KEY",  # the older name, still honoured
        "vision": True,
    },
    "OpenAI": {
        "kind": "openai",
        "model": "gpt-5-mini",
        "key_env": "OPENAI_API_KEY",
        "vision": True,
    },
    "OpenAI-style server": {
        "kind": "openai-compatible",
        "model": "gemma4:31b",  # needs about 20 GB of GPU memory; smaller ones garble arguments
        "base_url": "http://localhost:11434/v1",  # an Ollama or vLLM already running somewhere
        "vision": False,  # set by the operator in the window when their server can see
    },
}
DEFAULT_PROVIDER = "Gemini"
MODEL = "google:gemini-3.5-flash-lite"  # the model when no endpoint is chosen (tests, evals)
# The short names Pydantic AI uses in a "provider:model" string, by provider preset.
PREFIXES = {"google": "Gemini", "google-gla": "Gemini", "openai": "OpenAI"}
# Sampling and retries for every model, cloud or local. An agent that drives an
# instrument wants the most likely tool call, not a creative one, so the temperature
# is 0.
TEMPERATURE = 0.0
TOOL_CALL_RETRIES = 3  # a malformed tool call goes back to the model up to three times
MODEL_SETTINGS: dict[str, dict[str, Any]] = {
    # max_tokens: room for a full acquisition plan. parallel_tool_calls False: one
    # action at a time, so each is seen before the next.
    "google": {"temperature": TEMPERATURE, "max_tokens": 16000, "parallel_tool_calls": False},
    "openai": {"temperature": TEMPERATURE, "parallel_tool_calls": False},
    "openai-compatible": {"temperature": TEMPERATURE},  # small servers reject the parallel flag
}
DEFAULT_MODEL_SETTINGS = MODEL_SETTINGS["google"]  # the settings that go with MODEL

# -- a model file served on this computer (local.py) ------------------------------------
MODELS_FOLDER = "zmart_ai_agent_models"  # in the home folder, unless another is chosen
MODEL_SUFFIXES = (".gguf",)
# The context window the server is started with. llama-cpp-python's own default
# is 2,048 tokens, less than one request here (the instructions, the microscope's
# description, the tools and the state reading are about 7,000 tokens). 32K holds
# a request, a good number of turns, tool results and a margin.
CONTEXT_TOKENS = 32768
BATCH_TOKENS = 2048  # prompt batches: the long prefix is processed in fewer passes than at 512
FLASH_ATTENTION = True  # smaller memory and faster attention where the build supports it
SERVER_POLL_MS = 500  # how often the window asks whether the server is up
SERVER_START_TIMEOUT_S = 300  # a large file can take minutes to load from a slow disk

# -- the microscope --------------------------------------------------------------------
# The driver used when none is named: the simulated microscope that comes with the
# controller, which the controller's list of drivers always offers under this name.
DEFAULT_DRIVER = "mock"

# -- the tools -------------------------------------------------------------------------
# A stage move that travels further than this from where the stage was when the
# operator last wrote (on any one axis, in um) needs their go-ahead in the chat.
CONFIRM_XY_UM = 1000.0
CONFIRM_Z_UM = 100.0
# A procedure counts as a way to focus when its name holds this word ("autofocus",
# "find_focus"). Only the name counts: a description may mention focus in passing.
FOCUS_WORD = "focus"
# The start of a single look's position label, so its saved files are easy to tell apart.
LOOK_LABEL = "look"
# A plan's size, so a slip of the model cannot ask for a day of imaging.
PLAN_MAX_POSITIONS = 100
PLAN_MAX_TIME_POINTS = 1000
FILES_LISTED = 10  # saved files named in a run's answer; the rest are counted
WAIT_STEP_S = 0.1  # how often a run waiting for its next time point checks for Stop
# The camera image the vision model is shown is binned n x n first (each output pixel
# is the mean of an n x n block): 2 keeps a 2048-pixel camera image at 1024 pixels,
# enough for "is it centred" or "is it saturated", and averaging keeps dim detail
# that picking every other pixel would lose.
LOOK_BIN = 2
LOOK_MAX_SIDE = 1024  # a still larger image is binned further until it fits

# The source-reading tools.
SOURCE_MATCHES = 40  # search results returned at most
SOURCE_LINES = 200  # lines read at most in one go

# -- the frames (frames.py) --------------------------------------------------------------
# Every image a look or a run delivers is kept for the session as a small copy, its
# longer side at most FRAME_COPY_SIDE pixels, with its number, time, position, settings
# and the measured numbers. The oldest copies go once they take more than
# FRAME_HISTORY_BYTES together (about a hundred 256-pixel copies); the numbers keep
# counting. A look shows the eyes at most LOOK_FRAMES_MAX frames at once.
FRAME_COPY_SIDE = 256
FRAME_HISTORY_BYTES = 100 * 256 * 256 * 4
LOOK_FRAMES_MAX = 16
# The map, derived from the frames: a frame whose brightest pixel is less than
# MAP_SIGNAL_MIN of the camera's full range above the background shows no signal, and
# one with more than MAP_SATURATED_MAX percent of its pixels saturated is left out too.
# Frames within MAP_SAME_PLACE_UM of one another in x and y share a focus curve; where
# the sample sits is the median over the last MAP_PLACE_FRAMES usable frames.
MAP_SIGNAL_MIN = 0.003
MAP_SATURATED_MAX = 1.0
MAP_SAME_PLACE_UM = 25.0
MAP_PLACE_FRAMES = 5
# calibrate measures how the image moves when the stage moves (frames.Calibration) and
# keeps the answer per microscope and objective in this file under the computer's ZMART
# configuration folder. Its test move is CALIBRATE_STEP_FRACTION of the field of view
# (CALIBRATE_STEP_UM when the driver does not report its pixel size); a picture shift
# measured with less than CALIBRATE_CONFIDENCE_MIN confidence is no measurement.
CALIBRATION_FILE = ("zmart-ai-agent", "calibration.json")
CALIBRATE_STEP_FRACTION = 0.1
CALIBRATE_STEP_UM = 20.0
CALIBRATE_CONFIDENCE_MIN = 0.05

# -- the eyes ----------------------------------------------------------------------------
# The vision model keeps a conversation of its own for the session. A look attaches the
# frames it asks about; once answered, a turn keeps its words (the frames' numbers,
# measures, and what the eyes said) and loses the pictures, so a look costs the frames
# it shows and no more. The conversation is also cut to this many looks, oldest first,
# so a look every few minutes for a whole day does not send the whole day with every
# question.
VISION_TURNS_KEPT = 40

# -- schedules and requests --------------------------------------------------------------
# "Look every three minutes", "in ten minutes start the plan": the agent sets a
# schedule and the window's clock fires each due instruction as a turn of its own.
SCHEDULE_MIN_SECONDS = 5  # no schedule fires more often than this
SCHEDULES_MAX = 10
CLOCK_FORMAT = "%H:%M:%S"  # how the state, the eyes and the schedules write a time of day
# A request (requests.py) is what one typed message set going. Its "wait" leaves one
# continuation pending, for at most WAIT_MAX_S, and a request continues at most
# CONTINUATIONS_MAX times. A checklist in a reply is the request's plan, of at most
# PLAN_STEPS_MAX steps.
WAIT_MAX_S = 4 * 3600
CONTINUATIONS_MAX = 30
PLAN_STEPS_MAX = 12

# The conversation is made smaller now and then, between turns (see memory.compact()).
HISTORY_COMPACT_AFTER = 15  # operator turns before the history is made smaller
HISTORY_KEEP_TURNS = 10  # turns kept when it is; older ones are forgotten
HISTORY_FULL_TURNS = 3  # the newest turns keep their state readout and tool results in full
HISTORY_RESULT_CHARS = 300  # an older tool result is cut to this many characters

# -- the window --------------------------------------------------------------------------
FONT_POINTS = 11  # the window's letters: Qt's default of 9 is small at a microscope
