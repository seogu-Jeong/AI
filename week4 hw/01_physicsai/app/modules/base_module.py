"""
BaseModule — abstract base class for all simulation modules.
Implements the Template Method pattern (TRD-01 §2.3).

State machine: IDLE → TRAINING → TRAINED
                          ↓          ↓
                       STOPPED     DIRTY
                          ↓
                        ERROR → IDLE
"""
import time
from abc import abstractmethod
from enum import Enum, auto
from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QScrollArea, QPushButton, QLabel,
)
from PySide6.QtGui import QFont

from app.widgets.progress_panel import ProgressPanel
from app.ml.training_worker import TrainingWorker, TrainingConfig


class ModuleState(Enum):
    IDLE     = auto()
    TRAINING = auto()
    TRAINED  = auto()
    DIRTY    = auto()
    ERROR    = auto()


class BaseModule(QWidget):
    """Abstract base for all five simulation modules.

    Subclasses must implement:
        MODULE_ID, MODULE_NAME, MODULE_DESC
        _setup_param_panel() → QWidget
        _setup_plot_area()   → QWidget
        _build_model()
        _generate_data()     → (X, y)
        _get_training_config() → TrainingConfig
        _on_training_finished(model, history)
        get_param_values()   → dict
        get_metrics()        → dict
    """

    MODULE_ID:   str = "MOD-XX"
    MODULE_NAME: str = "Module"
    MODULE_DESC: str = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state   : ModuleState        = ModuleState.IDLE
        self._worker  : Optional[TrainingWorker] = None
        self._model                         = None
        self._start_ms: float               = 0.0

        # Public training metrics (for export)
        self.epochs_run          : Optional[int]   = None
        self.final_loss          : Optional[float] = None
        self.final_val_loss      : Optional[float] = None
        self.training_elapsed_ms : Optional[int]   = None

        self._setup_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # Layout assembly
    # ─────────────────────────────────────────────────────────────────────────
    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel (parameter controls) ──────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(300)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        # Module title badge
        title_lbl = QLabel(self.MODULE_NAME)
        f = QFont(); f.setPointSize(13); f.setBold(True)
        title_lbl.setFont(f)
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title_lbl)

        if self.MODULE_DESC:
            desc_lbl = QLabel(self.MODULE_DESC)
            desc_lbl.setWordWrap(True)
            desc_lbl.setStyleSheet("color: #757575; font-size: 9pt;")
            desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            left_layout.addWidget(desc_lbl)

        # Subclass parameter panel
        self._param_panel = self._setup_param_panel()
        left_layout.addWidget(self._param_panel)

        # ── Control buttons ───────────────────────────────────────────────────
        self._run_btn   = QPushButton("▶  Run")
        self._stop_btn  = QPushButton("■  Stop")
        self._reset_btn = QPushButton("↺  Reset")
        for btn, obj_name in [(self._run_btn,   'run_btn'),
                               (self._stop_btn,  'stop_btn'),
                               (self._reset_btn, 'reset_btn')]:
            btn.setObjectName(obj_name)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addWidget(self._reset_btn)
        left_layout.addLayout(btn_row)

        # ── Progress panel ────────────────────────────────────────────────────
        self._progress = ProgressPanel()
        left_layout.addWidget(self._progress)

        # ── Status label ──────────────────────────────────────────────────────
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet("font-size: 9pt; color: #9E9E9E;")
        left_layout.addWidget(self._status_lbl)
        left_layout.addStretch()

        left_scroll.setWidget(left_inner)
        splitter.addWidget(left_scroll)

        # ── Right panel (plots) ───────────────────────────────────────────────
        plot_area = self._setup_plot_area()
        splitter.addWidget(plot_area)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])

        root.addWidget(splitter)

        # ── Wire signals ──────────────────────────────────────────────────────
        self._run_btn.clicked.connect(self.run)
        self._stop_btn.clicked.connect(self.stop)
        self._reset_btn.clicked.connect(self.reset)
        self._apply_state(ModuleState.IDLE)

    # ─────────────────────────────────────────────────────────────────────────
    # Abstract interface
    # ─────────────────────────────────────────────────────────────────────────
    @abstractmethod
    def _setup_param_panel(self) -> QWidget:
        ...

    @abstractmethod
    def _setup_plot_area(self) -> QWidget:
        ...

    @abstractmethod
    def _build_model(self):
        ...

    @abstractmethod
    def _generate_data(self):
        ...

    @abstractmethod
    def _get_training_config(self) -> TrainingConfig:
        ...

    @abstractmethod
    def _on_training_finished_impl(self, model, history):
        ...

    def get_param_values(self) -> Dict[str, Any]:
        return {}

    def get_metrics(self) -> Dict[str, Any]:
        return {}

    # ─────────────────────────────────────────────────────────────────────────
    # Template method — run()
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        if self._state == ModuleState.TRAINING:
            return
        try:
            model  = self._build_model()
            data   = self._generate_data()
            config = self._get_training_config()
        except Exception as exc:
            self._apply_state(ModuleState.ERROR)
            self._status_lbl.setText(str(exc))
            return

        X, y = data[0], data[1]
        self._worker = TrainingWorker(model, X, y, config, parent=self)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.training_finished.connect(self._on_training_finished)
        self._worker.training_error.connect(self._on_training_error)

        self._start_ms = time.monotonic()
        self._progress.start(config.epochs)
        self._apply_state(ModuleState.TRAINING)
        self._worker.start()

    def stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
        self._apply_state(ModuleState.IDLE)
        self._status_lbl.setText("Stopped")

    def reset(self):
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(2000)
        self._model = None
        self._progress.reset()
        self._apply_state(ModuleState.IDLE)
        self._status_lbl.setText("Ready")
        self._on_reset_impl()

    def _on_reset_impl(self):
        """Override to add module-specific reset logic."""
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # Worker signal slots (main thread)
    # ─────────────────────────────────────────────────────────────────────────
    def _on_progress(self, epoch: int, loss: float, val_loss: float):
        self._progress.update(epoch, self._get_training_config().epochs, loss, val_loss)

    def _on_training_finished(self, model, history):
        self._model = model
        elapsed_ms = int((time.monotonic() - self._start_ms) * 1000)

        # Record metrics
        h = history.history
        self.epochs_run          = len(h.get('loss', []))
        self.final_loss          = float(h['loss'][-1]) if 'loss' in h else None
        self.final_val_loss      = float(h['val_loss'][-1]) if 'val_loss' in h else None
        self.training_elapsed_ms = elapsed_ms

        self._progress.complete()
        self._apply_state(ModuleState.TRAINED)
        self._status_lbl.setText(f"Trained · {elapsed_ms:,} ms")
        self._on_training_finished_impl(model, history)

    def _on_training_error(self, msg: str):
        self._apply_state(ModuleState.ERROR)
        self._progress.error(msg)
        self._status_lbl.setText("Error — reset to try again")

    # ─────────────────────────────────────────────────────────────────────────
    # State machine
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_state(self, state: ModuleState):
        self._state = state
        run_ok   = state in (ModuleState.IDLE, ModuleState.TRAINED, ModuleState.DIRTY)
        stop_ok  = state == ModuleState.TRAINING
        reset_ok = state in (ModuleState.TRAINED, ModuleState.DIRTY, ModuleState.ERROR)

        self._run_btn.setEnabled(run_ok)
        self._stop_btn.setEnabled(stop_ok)
        self._reset_btn.setEnabled(reset_ok)

        dirty = state == ModuleState.DIRTY
        self._run_btn.setProperty("dirty", "true" if dirty else "false")
        self._run_btn.style().unpolish(self._run_btn)
        self._run_btn.style().polish(self._run_btn)

    def _mark_dirty(self):
        """Call when params change post-training."""
        if self._state == ModuleState.TRAINED:
            self._apply_state(ModuleState.DIRTY)

    # ─────────────────────────────────────────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────────────────────────────────────────
    def cleanup(self):
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(2000)
