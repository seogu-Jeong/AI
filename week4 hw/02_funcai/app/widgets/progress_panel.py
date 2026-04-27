"""ProgressPanel — training progress bar + epoch/loss/time labels."""
import time
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                QLabel, QProgressBar)
from PySide6.QtGui import QFont


class ProgressPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._start: Optional[float] = None
        self._total = 0
        self._setup_ui()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4); lay.setSpacing(3)
        self._bar = QProgressBar(); self._bar.setValue(0)
        lay.addWidget(self._bar)
        row1 = QHBoxLayout()
        self._epoch_lbl = QLabel("Epoch: —")
        self._loss_lbl  = QLabel("Loss: —")
        bold = QFont(); bold.setBold(True); self._loss_lbl.setFont(bold)
        row1.addWidget(self._epoch_lbl); row1.addStretch(); row1.addWidget(self._loss_lbl)
        lay.addLayout(row1)
        row2 = QHBoxLayout()
        self._vloss_lbl = QLabel("Val Loss: —")
        self._time_lbl  = QLabel("")
        self._time_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        row2.addWidget(self._vloss_lbl); row2.addStretch(); row2.addWidget(self._time_lbl)
        lay.addLayout(row2)

    def start(self, total: int):
        self._total = total; self._bar.setRange(0, total); self._bar.setValue(0)
        self._start = time.monotonic(); self._time_lbl.setText("Training…")
        for lbl in (self._epoch_lbl, self._loss_lbl, self._vloss_lbl, self._time_lbl):
            lbl.setStyleSheet("")

    def update(self, epoch: int, total: int, loss: float, val_loss: float):
        self._bar.setValue(epoch)
        self._epoch_lbl.setText(f"Epoch: {epoch:,} / {total:,}")
        self._loss_lbl.setText(f"Loss: {loss:.6f}")
        if val_loss != loss:
            self._vloss_lbl.setText(f"Val: {val_loss:.6f}")
        if self._start:
            self._time_lbl.setText(f"{time.monotonic()-self._start:.1f}s")

    def complete(self):
        self._bar.setValue(self._bar.maximum())
        if self._start:
            ms = int((time.monotonic() - self._start) * 1000)
            self._time_lbl.setText(f"Done {ms:,} ms")
        self._loss_lbl.setStyleSheet("color: #43A047; font-weight: bold;")

    def error(self, msg: str):
        self._time_lbl.setText("Error"); self._time_lbl.setStyleSheet("color:#E53935;")
        self._epoch_lbl.setText(msg[:60])

    def reset(self):
        self._bar.setValue(0)
        for lbl in (self._epoch_lbl, self._loss_lbl, self._vloss_lbl, self._time_lbl):
            lbl.setText("—" if lbl is not self._time_lbl else "")
            lbl.setStyleSheet("")
