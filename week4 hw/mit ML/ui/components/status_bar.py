from __future__ import annotations
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainter, QColor

from ui.theme import BG, WHITE_60


class StatusBar(QWidget):
    """Single-line status display with glassmorphism background."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self._text = "Ready"
        self._label = QLabel(self._text, self)
        self._label.setStyleSheet(
            "color: rgba(255,255,255,0.7); font-size: 9pt; background: transparent;"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.addWidget(self._label)

    def set_text(self, text: str) -> None:
        self._text = text
        self._label.setText(text)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(255, 255, 255, 12))
        p.setPen(QColor(255, 255, 255, 20))
        p.drawLine(0, 0, self.width(), 0)
        p.end()
