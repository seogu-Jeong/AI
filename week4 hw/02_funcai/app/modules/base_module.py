"""
BaseModule — abstract base for MOD-01 and MOD-03.
Implements Template Method pattern; subclasses override abstract methods.
State machine: IDLE → TRAINING → TRAINED / DIRTY / ERROR
"""
import time
from abc import abstractmethod
from enum import Enum, auto
from typing import Optional, Dict, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
                                QScrollArea, QPushButton, QLabel)
from PySide6.QtGui import QFont

from app.widgets.progress_panel import ProgressPanel
from app.ml.training_worker import TrainingWorker, TrainingConfig


class ModuleState(Enum):
    IDLE = auto(); TRAINING = auto(); TRAINED = auto()
    DIRTY = auto(); ERROR = auto()


class BaseModule(QWidget):
    MODULE_ID:   str = "MOD-XX"
    MODULE_NAME: str = "Module"
    MODULE_DESC: str = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state:  ModuleState           = ModuleState.IDLE
        self._worker: Optional[TrainingWorker] = None
        self._model                          = None
        self._start_ms: float               = 0.0
        self.epochs_run:          Optional[int]   = None
        self.final_loss:          Optional[float] = None
        self.final_val_loss:      Optional[float] = None
        self.training_elapsed_ms: Optional[int]   = None
        self._setup_ui()

    def _setup_ui(self):
        root = QHBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFixedWidth(310)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(); lay = QVBoxLayout(inner)
        lay.setContentsMargins(10,10,10,10); lay.setSpacing(8)

        title = QLabel(self.MODULE_NAME)
        f = QFont(); f.setPointSize(13); f.setBold(True); title.setFont(f)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter); lay.addWidget(title)

        if self.MODULE_DESC:
            desc = QLabel(self.MODULE_DESC); desc.setWordWrap(True)
            desc.setStyleSheet("color:#757575;font-size:9pt;")
            desc.setAlignment(Qt.AlignmentFlag.AlignCenter); lay.addWidget(desc)

        self._param_panel = self._setup_param_panel(); lay.addWidget(self._param_panel)

        self._run_btn   = QPushButton("▶  Run")
        self._stop_btn  = QPushButton("■  Stop")
        self._reset_btn = QPushButton("↺  Reset")
        for btn, name in [(self._run_btn,'run_btn'),(self._stop_btn,'stop_btn'),(self._reset_btn,'reset_btn')]:
            btn.setObjectName(name)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._run_btn); btn_row.addWidget(self._stop_btn); btn_row.addWidget(self._reset_btn)
        lay.addLayout(btn_row)

        self._progress = ProgressPanel(); lay.addWidget(self._progress)
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet("font-size:9pt;color:#9E9E9E;")
        lay.addWidget(self._status_lbl); lay.addStretch()
        scroll.setWidget(inner); splitter.addWidget(scroll)

        plot_area = self._setup_plot_area(); splitter.addWidget(plot_area)
        splitter.setStretchFactor(0,0); splitter.setStretchFactor(1,1)
        splitter.setSizes([310, 900]); root.addWidget(splitter)

        self._run_btn.clicked.connect(self.run)
        self._stop_btn.clicked.connect(self.stop)
        self._reset_btn.clicked.connect(self.reset)
        self._apply_state(ModuleState.IDLE)

    @abstractmethod
    def _setup_param_panel(self) -> QWidget: ...
    @abstractmethod
    def _setup_plot_area(self) -> QWidget: ...
    @abstractmethod
    def _build_model(self): ...
    @abstractmethod
    def _generate_data(self): ...
    @abstractmethod
    def _get_training_config(self) -> TrainingConfig: ...
    @abstractmethod
    def _on_training_finished_impl(self, model, history): ...

    def get_param_values(self) -> Dict[str, Any]: return {}
    def get_metrics(self) -> Dict[str, Any]: return {}

    def run(self):
        if self._state == ModuleState.TRAINING: return
        try:
            model = self._build_model(); data = self._generate_data()
            config = self._get_training_config()
        except Exception as exc:
            self._apply_state(ModuleState.ERROR)
            self._status_lbl.setText(str(exc)); return
        X, y = data[0], data[1]
        xv = data[2] if len(data) > 2 else None
        yv = data[3] if len(data) > 3 else None
        self._worker = TrainingWorker(model, X, y, config, X_val=xv, y_val=yv, parent=self)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.training_finished.connect(self._on_training_finished)
        self._worker.training_error.connect(self._on_training_error)
        self._start_ms = time.monotonic()
        self._progress.start(config.epochs)
        self._apply_state(ModuleState.TRAINING); self._worker.start()

    def stop(self):
        if self._worker and self._worker.isRunning(): self._worker.request_stop()
        self._apply_state(ModuleState.IDLE); self._status_lbl.setText("Stopped")

    def reset(self):
        if self._worker and self._worker.isRunning():
            self._worker.request_stop(); self._worker.wait(2000)
        self._model = None; self._progress.reset()
        self._apply_state(ModuleState.IDLE); self._status_lbl.setText("Ready")
        self._on_reset_impl()

    def _on_reset_impl(self): pass

    def _on_progress(self, epoch, loss, val_loss):
        self._progress.update(epoch, self._get_training_config().epochs, loss, val_loss)

    def _on_training_finished(self, model, history):
        self._model = model
        elapsed_ms = int((time.monotonic() - self._start_ms) * 1000)
        h = history.history
        self.epochs_run          = len(h.get('loss', []))
        self.final_loss          = float(h['loss'][-1]) if 'loss' in h else None
        self.final_val_loss      = float(h['val_loss'][-1]) if 'val_loss' in h else None
        self.training_elapsed_ms = elapsed_ms
        self._progress.complete()
        self._apply_state(ModuleState.TRAINED)
        self._status_lbl.setText(f"Trained · {elapsed_ms:,} ms")
        self._on_training_finished_impl(model, history)

    def _on_training_error(self, msg):
        self._apply_state(ModuleState.ERROR)
        self._progress.error(msg); self._status_lbl.setText("Error — reset to retry")

    def _apply_state(self, state: ModuleState):
        self._state = state
        self._run_btn.setEnabled(state in (ModuleState.IDLE, ModuleState.TRAINED, ModuleState.DIRTY))
        self._stop_btn.setEnabled(state == ModuleState.TRAINING)
        self._reset_btn.setEnabled(state in (ModuleState.TRAINED, ModuleState.DIRTY, ModuleState.ERROR))
        dirty = state == ModuleState.DIRTY
        self._run_btn.setProperty("dirty", "true" if dirty else "false")
        self._run_btn.style().unpolish(self._run_btn)
        self._run_btn.style().polish(self._run_btn)

    def _mark_dirty(self):
        if self._state == ModuleState.TRAINED: self._apply_state(ModuleState.DIRTY)

    def cleanup(self):
        if self._worker and self._worker.isRunning():
            self._worker.request_stop(); self._worker.wait(2000)
