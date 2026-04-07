from __future__ import annotations
from PySide6.QtCore import QObject, Signal, QTimer


class ConceptController(QObject):
    """Drives animated concept demos (phase animation)."""

    phase_updated = Signal(float)   # 0.0 … 1.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._phase = 0.0
        self._speed = 0.01

    def start(self, speed: float = 0.01):
        self._speed = speed
        self._timer.start(33)

    def stop(self):
        self._timer.stop()

    def _tick(self):
        self._phase = (self._phase + self._speed) % 1.0
        self.phase_updated.emit(self._phase)
