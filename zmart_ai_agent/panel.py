"""The Model panel at the top of the chat window.

A single line shows which model is in use; clicking it folds the panel open.
Inside are two boxes: the language model that runs the conversation, and the
vision model that is shown camera images (by default the same one). Each box
names either a cloud model (provider, model, API key) or a model file on this
computer, which the window serves itself (see ``local.py``). "Use this model"
applies the choice; the conversation so far is kept. Below them, the
Preferences box sets the size of the letters. Where images are saved is the
microscope driver's choice, so it is not set here.

The API key typed here stays in memory for this session only.

Author: Thom de Hoog, Center for Microscopy and Image Analysis (ZMB), University of Zurich
        thom.dehoog@zmb.uzh.ch . thomdehoog@gmail.com
Date: 2026-10-02
License: MIT
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import local, models
from .models import Endpoint
from .settings import (
    DEFAULT_PROVIDER,
    PROVIDERS,
    SERVER_POLL_MS,
    SERVER_START_TIMEOUT_S,
)

CLOUD = "Cloud"
LOCAL_FILE = "File on this computer"
SAME = "Same as language model"
LOCAL_PROVIDER = "Local file"


class ModelPicker(QGroupBox):
    """A titled box that names one model.

    The Type dropdown decides the fields after it: for a cloud model the provider,
    model name, API key and (for an OpenAI-style server) its address; for a file
    on this computer the file and its folder. The vision box has one more type,
    "Same as language model".
    """

    changed = Signal()

    def __init__(self, title: str, same_as: bool = False, models_folder: Path | None = None):
        super().__init__(title)
        self.models_folder = Path(models_folder or local.DEFAULT_FOLDER)
        grid = QGridLayout(self)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self.grid = grid

        self.mode = QComboBox()
        self.mode.addItems(([SAME] if same_as else []) + [CLOUD, LOCAL_FILE])
        self.provider = QComboBox()
        self.provider.addItems(list(PROVIDERS))
        self.provider.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.model = QLineEdit()
        self.model.setMinimumWidth(140)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setMinimumWidth(120)
        self.base_url = QLineEdit()
        self.base_url.setMinimumWidth(160)
        self.sees = QCheckBox("Can see images")
        self.sees.setToolTip(
            "Tick when this server's model can look at pictures, so the assistant "
            "shows it the camera image. Untick for a text-only model."
        )
        self.local_model = QComboBox()
        self.local_model.setMinimumWidth(160)
        self.folder_button = QPushButton("Models folder ...")

        self.labels = {
            name: QLabel(text)
            for name, text in {
                "mode": "Type",
                "provider": "Provider",
                "model": "Model",
                "key": "API key",
                "base_url": "Address",
                "local_model": "File",
            }.items()
        }
        # Line one: the type and what names the model. Line two, cloud only: the key,
        # and for a server its address. The model column takes the leftover width.
        grid.addWidget(self.labels["mode"], 0, 0)
        grid.addWidget(self.mode, 0, 1)
        grid.addWidget(self.labels["provider"], 0, 2)
        grid.addWidget(self.provider, 0, 3)
        grid.addWidget(self.labels["model"], 0, 4)
        grid.addWidget(self.model, 0, 5, 1, 3)
        grid.addWidget(self.labels["local_model"], 0, 2)
        grid.addWidget(self.local_model, 0, 3, 1, 4)
        grid.addWidget(self.folder_button, 0, 7)
        grid.addWidget(self.labels["key"], 1, 2)
        grid.addWidget(self.key, 1, 3, 1, 2)
        grid.addWidget(self.labels["base_url"], 1, 5)
        grid.addWidget(self.base_url, 1, 6)
        grid.addWidget(self.sees, 1, 7)
        grid.setColumnStretch(5, 1)

        self.provider.setCurrentText(DEFAULT_PROVIDER)
        self.mode.setCurrentText(SAME if same_as else CLOUD)
        self.mode.currentTextChanged.connect(self._on_mode_changed)
        self.provider.currentTextChanged.connect(self._on_provider_changed)
        self.folder_button.clicked.connect(self.choose_folder)
        for widget in (self.mode, self.provider, self.local_model):
            widget.currentTextChanged.connect(lambda *_: self.changed.emit())
        self._on_provider_changed(DEFAULT_PROVIDER)

    # -- what is chosen -------------------------------------------------------------------

    @property
    def same(self) -> bool:
        """True when this box defers to the language model (the vision box's first choice)."""
        return self.mode.currentText() == SAME

    @property
    def is_local(self) -> bool:
        return self.mode.currentText() == LOCAL_FILE

    def cloud_endpoint(self) -> Endpoint:
        provider = self.provider.currentText()
        kind = PROVIDERS[provider]["kind"]
        vision = self.sees.isChecked() if kind == "openai-compatible" else None
        return Endpoint.from_preset(
            provider, self.model.text(), self.key.text(), self.base_url.text(), vision=vision
        )

    def local_path(self) -> Path | None:
        name = self.local_model.currentText()
        return self.models_folder / name if name else None

    def show_endpoint(self, endpoint: Endpoint) -> None:
        """Fill the fields from an endpoint, e.g. the one named on the command line."""
        self.mode.setCurrentText(CLOUD)
        self.provider.setCurrentText(endpoint.provider)
        self.model.setText(endpoint.model)
        if endpoint.kind == "openai-compatible":
            self.base_url.setText(endpoint.base_url)
            self.sees.setChecked(endpoint.vision)

    # -- the fields ----------------------------------------------------------------------

    def _on_mode_changed(self, *_) -> None:
        """Show the fields for the type chosen; the file list is rescanned when it appears."""
        cloud = self.mode.currentText() == CLOUD
        is_local = self.is_local
        server = cloud and PROVIDERS[self.provider.currentText()]["kind"] == ("openai-compatible")
        for widget in (self.labels["local_model"], self.local_model, self.folder_button):
            widget.setVisible(is_local)
        for name in ("provider", "model", "key"):
            self.labels[name].setVisible(cloud)
            getattr(self, name).setVisible(cloud)
        for widget in (self.labels["base_url"], self.base_url, self.sees):
            widget.setVisible(server)
        if is_local:
            self.scan_models()

    def _on_provider_changed(self, name: str) -> None:
        """Prefill the preset and say where the key comes from when the field is left empty."""
        preset = PROVIDERS[name]
        self.model.setText(preset["model"])
        self.base_url.setText(preset.get("base_url", ""))
        variable = preset.get("key_env")
        in_env = bool(variable and os.environ.get(variable))
        if preset["kind"] == "openai-compatible":
            placeholder = "only if the server asks for one"
        elif in_env:
            placeholder = f"using {variable} from the environment"
        else:
            placeholder = f"{name} API key"
        self.key.setPlaceholderText(placeholder)
        self._on_mode_changed()

    def scan_models(self) -> None:
        """Fill the file dropdown from the models folder; an empty folder leaves it greyed."""
        current = self.local_model.currentText()
        names = local.list_models(self.models_folder)
        self.local_model.clear()
        self.local_model.addItems(names)
        if current in names:
            self.local_model.setCurrentText(current)
        self.local_model.setEnabled(bool(names))
        self.local_model.setToolTip(
            f"Model files (.gguf) in {self.models_folder}"
            if names
            else f"No .gguf model files in {self.models_folder}. Choose another folder, or put "
            "model files there."
        )

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Models folder", str(self.models_folder))
        if folder:
            self.models_folder = Path(folder)
            self.scan_models()


class PreferencesBox(QGroupBox):
    """The third box: the size of the letters. ``on_font`` gets the point size, at once."""

    def __init__(self, font_points: int, on_font: Callable[[int], None]) -> None:
        super().__init__("Preferences")
        self.on_font = on_font
        self.font_size = QSpinBox()
        self.font_size.setRange(8, 24)
        self.font_size.setSuffix(" pt")
        self.font_size.setValue(font_points)
        self.font_size.valueChanged.connect(self.on_font)
        grid = QGridLayout(self)
        grid.setHorizontalSpacing(8)
        grid.addWidget(QLabel("Letter size"), 0, 0)
        grid.addWidget(self.font_size, 0, 1)
        grid.setColumnStretch(2, 1)


class ModelPanel(QWidget):
    """One line that names the model in use, folding open to the pickers and the preferences.

    ``apply`` (also the "Use this model" button) builds the endpoints and hands
    them to ``on_use``. A model file is served first; the panel waits for the
    server and calls ``on_use`` when it answers.
    """

    def __init__(
        self,
        on_use: Callable[[Endpoint, Endpoint | None], None],
        say: Callable[[str], None],
        preferences: PreferencesBox | None = None,
        models_folder: Path | None = None,
    ) -> None:
        super().__init__()
        self.on_use = on_use
        self.say = say
        self.preferences = preferences
        self.servers: list[local.LocalModelServer] = []
        self._pending: dict | None = None

        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setChecked(False)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setStyleSheet("QToolButton { border: none; font-weight: bold }")
        self.toggle.toggled.connect(self._fold)
        self.summary = QLabel("no model chosen yet")
        self.summary.setStyleSheet("color:#555")
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.toggle)
        header.addWidget(self.summary, 1)

        self.language = ModelPicker("Language model", models_folder=models_folder)
        self.vision = ModelPicker("Vision model", same_as=True, models_folder=models_folder)
        self.use_button = QPushButton("Use this model", clicked=self.apply)
        self.note = QLabel(wordWrap=True)
        self.note.setStyleSheet("color:#555")
        buttons = QHBoxLayout()
        buttons.addWidget(self.use_button)
        buttons.addWidget(self.note, 1)
        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.language)
        body.addWidget(self.vision)
        body.addLayout(buttons)
        if preferences is not None:
            body.addWidget(preferences)
        self.body = QWidget()
        self.body.setLayout(body)
        self.body.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addWidget(self.body)
        self._fold(False)

    def _fold(self, open_: bool) -> None:
        self.body.setVisible(open_)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow)
        self.toggle.setText("Model" if open_ else "Model:")

    # -- applying a choice -----------------------------------------------------------------

    def apply(self) -> None:
        """Use what the pickers say. A model file is served first, then used when it answers."""
        self.stop_servers()
        self._pending = {}
        self.note.setText("")
        try:
            for role, picker in (("language", self.language), ("vision", self.vision)):
                if picker.same:
                    self._pending[role] = None
                elif picker.is_local:
                    path = picker.local_path()
                    if path is None:
                        raise RuntimeError(f"choose a model file for the {picker.title().lower()}")
                    self._pending[role] = self._serve(path)
                else:
                    endpoint = picker.cloud_endpoint()
                    if endpoint.needs_key and not endpoint.api_key:
                        raise RuntimeError(models.missing_key_advice(endpoint))
                    self._pending[role] = endpoint
        except (RuntimeError, OSError) as exc:
            self.stop_servers()
            self._pending = None
            self.note.setText(str(exc))
            self.say(f"The model was not changed: {exc}")
            self.toggle.setChecked(True)  # open, so the operator sees what to fill in
            return
        if self.servers:
            self.use_button.setEnabled(False)
            self.note.setText("Loading the model file ... (a large file takes a while)")
            self._waited_ms = 0
            QTimer.singleShot(SERVER_POLL_MS, self._poll)
        else:
            self._finish()

    def _serve(self, path: Path) -> local.LocalModelServer:
        projector = local.projector_for(path.parent, path.name)
        server = local.LocalModelServer(str(path), projector=projector)
        server.start()
        self.servers.append(server)
        return server

    def _poll(self) -> None:
        if self._pending is None:
            return
        try:
            ready = all(s.ready() for s in self.servers)
        except RuntimeError as exc:
            self._failed(str(exc))
            return
        self._waited_ms += SERVER_POLL_MS
        if not ready:
            if self._waited_ms > SERVER_START_TIMEOUT_S * 1000:
                self._failed("the model server did not answer in time")
            else:
                QTimer.singleShot(SERVER_POLL_MS, self._poll)
            return
        self._finish()

    def _failed(self, why: str) -> None:
        self.stop_servers()
        self._pending = None
        self.use_button.setEnabled(True)
        self.note.setText(why)
        self.say(f"The model file could not be served: {why}")

    def _finish(self) -> None:
        pending, self._pending = self._pending, None
        endpoints = {}
        for role, item in pending.items():
            if isinstance(item, local.LocalModelServer):
                item = Endpoint(
                    provider=LOCAL_PROVIDER,
                    kind="openai-compatible",
                    model=item.model,
                    base_url=item.base_url,
                    vision=item.projector is not None,
                )
            endpoints[role] = item
        self.use_button.setEnabled(True)
        language, vision = endpoints["language"], endpoints["vision"]
        try:
            self.on_use(language, vision)
        except Exception as exc:  # a provider library that is not installed, most likely
            self.note.setText(str(exc))
            self.say(f"The model was not changed: {exc}")
            return
        seeing = vision or language
        sight = "" if seeing.vision else " (cannot see images)"
        self.summary.setText(language.name + (f", vision {vision.name}" if vision else "") + sight)
        self.note.setText("In use." if language.needs_key or language.base_url else "")
        self.toggle.setChecked(False)

    def stop_servers(self) -> None:
        for server in self.servers:
            server.stop()
        self.servers.clear()
