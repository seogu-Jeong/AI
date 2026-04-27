from __future__ import annotations
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QRectF
from PySide6.QtGui import QPainter

from ui.visualizations.painter_utils import draw_glass_rect


class GlassPanel(QWidget):
    """Glassmorphism container widget.

    Do NOT call QVBoxLayout(panel) externally — use panel.layout() instead.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        draw_glass_rect(p, QRectF(1, 1, self.width() - 2, self.height() - 2))
        p.end()
