"""
ProgressPanel — training progress display.
Epoch counter, loss metrics, elapsed time.
"""
import time
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar,
)
from PySide6.QtGui import QFont


class ProgressPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._start_time: Optional[float] = None
        self._total_epochs = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        # Progress bar
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFormat("%p%")
        layout.addWidget(self._bar)

        # Epoch / loss row
        row1 = QHBoxLayout()
        self._epoch_lbl = QLabel("Epoch: —")
        self._loss_lbl  = QLabel("Loss: —")
        bold = QFont(); bold.setBold(True)
        self._loss_lbl.setFont(bold)
        row1.addWidget(self._epoch_lbl)
        row1.addStretch()
        row1.addWidget(self._loss_lbl)
        layout.addLayout(row1)

        # Val loss / time row
        row2 = QHBoxLayout()
        self._vloss_lbl = QLabel("Val Loss: —")
        self._time_lbl  = QLabel("")
        self._time_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        row2.addWidget(self._vloss_lbl)
        row2.addStretch()
        row2.addWidget(self._time_lbl)
        layout.addLayout(row2)

    def start(self, total_epochs: int):
        self._total_epochs = total_epochs
        self._bar.setRange(0, total_epochs)
        self._bar.setValue(0)
        self._start_time = time.monotonic()
        self._time_lbl.setText("Training…")
        self._time_lbl.setStyleSheet("")
        self._loss_lbl.setStyleSheet("")
        for lbl in (self._epoch_lbl, self._loss_lbl, self._vloss_lbl):
            lbl.setStyleSheet("")

    def update(self, epoch: int, total: int, loss: float, val_loss: float):
        self._bar.setValue(epoch)
        self._epoch_lbl.setText(f"Epoch: {epoch:,} / {total:,}")
        self._loss_lbl.setText(f"Loss: {loss:.6f}")
        if val_loss:
            self._vloss_lbl.setText(f"Val Loss: {val_loss:.6f}")
        if self._start_time:
            elapsed = time.monotonic() - self._start_time
            self._time_lbl.setText(f"{elapsed:.1f}s")

    def complete(self):
        self._bar.setValue(self._bar.maximum())
        if self._start_time:
            ms = int((time.monotonic() - self._start_time) * 1000)
            self._time_lbl.setText(f"Done in {ms:,} ms")
        self._loss_lbl.setStyleSheet("color: #43A047; font-weight: bold;")

    def error(self, message: str):
        self._time_lbl.setText("Error")
        self._time_lbl.setStyleSheet("color: #E53935; font-weight: bold;")
        self._epoch_lbl.setText(message[:60])

    def reset(self):
        self._bar.setValue(0)
        for lbl in (self._epoch_lbl, self._loss_lbl, self._vloss_lbl, self._time_lbl):
            lbl.setText("—" if lbl is not self._time_lbl else "")
            lbl.setStyleSheet("")
