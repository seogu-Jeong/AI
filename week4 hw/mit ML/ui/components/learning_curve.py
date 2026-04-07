from __future__ import annotations
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QRectF, QPointF, Qt
from PySide6.QtGui import QPainter, QColor, QPen, QLinearGradient, QPolygonF, QFont

from ui.theme import CYAN, WHITE_40, BG


class LearningCurve(QWidget):
    """Live scrolling reward chart drawn with QPainter."""

    def __init__(self, max_points: int = 200, line_color: QColor = CYAN, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._data: list[float] = []
        self._max_points = max_points
        self._line_color = line_color

    def add_point(self, value: float) -> None:
        self._data.append(value)
        if len(self._data) > self._max_points:
            self._data.pop(0)
        self.update()

    def clear(self) -> None:
        self._data.clear(); self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        if len(self._data) < 2:
            p.setPen(WHITE_40)
            font = QFont(); font.setPointSize(8); p.setFont(font)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No data yet")
            p.end(); return

        W, H = self.width(), self.height()
        pad_x, pad_y = 36, 12
        chart_w = W - pad_x - 8; chart_h = H - 2 * pad_y

        min_v = min(self._data); max_v = max(self._data)
        span = max(max_v - min_v, 1.0)

        def to_screen(i, v):
            sx = pad_x + (i / max(len(self._data) - 1, 1)) * chart_w
            sy = pad_y + (1 - (v - min_v) / span) * chart_h
            return sx, sy

        # Gradient fill under curve
        g = QLinearGradient(0, pad_y, 0, pad_y + chart_h)
        g.setColorAt(0, QColor(self._line_color.red(),
                               self._line_color.green(),
                               self._line_color.blue(), 60))
        g.setColorAt(1, QColor(self._line_color.red(),
                               self._line_color.green(),
                               self._line_color.blue(), 0))

        poly_pts = [QPointF(pad_x, pad_y + chart_h)]
        for i, v in enumerate(self._data):
            sx, sy = to_screen(i, v)
            poly_pts.append(QPointF(sx, sy))
        poly_pts.append(QPointF(pad_x + chart_w, pad_y + chart_h))
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(g)
        p.drawPolygon(QPolygonF(poly_pts))

        # Line
        p.setPen(QPen(self._line_color, 1.5))
        for i in range(1, len(self._data)):
            x1, y1 = to_screen(i - 1, self._data[i - 1])
            x2, y2 = to_screen(i, self._data[i])
            p.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Y-axis labels
        font = QFont(); font.setPointSize(7); p.setFont(font)
        p.setPen(WHITE_40)
        for v in [min_v, (min_v + max_v) / 2, max_v]:
            _, sy = to_screen(0, v)
            p.drawText(QRectF(0, sy - 8, pad_x - 4, 16),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{v:.1f}")

        p.end()
