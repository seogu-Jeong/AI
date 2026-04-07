from __future__ import annotations
import math
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QRectF, QPointF, Qt
from PySide6.QtGui import QPainter, QColor, QPen, QLinearGradient, QFont

from ui.theme import CYAN, MAGENTA, EMERALD, AMBER, BG, WHITE_60
from ui.visualizations.painter_utils import draw_glow


class CartPoleView(QWidget):
    """Renders CartPole physics: cart, pole, track, state bars."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 240)
        self._x = 0.0
        self._theta = 0.0
        self._x_dot = 0.0
        self._theta_dot = 0.0
        self._done = False

    def update_state(self, x: float, x_dot: float,
                     theta: float, theta_dot: float, done: bool = False) -> None:
        self._x = x; self._x_dot = x_dot
        self._theta = theta; self._theta_dot = theta_dot
        self._done = done
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        track_y = H * 0.65
        track_x0 = W * 0.1; track_x1 = W * 0.9
        track_w = track_x1 - track_x0

        p.setPen(QPen(QColor(255, 255, 255, 40), 2))
        p.drawLine(int(track_x0), int(track_y), int(track_x1), int(track_y))

        cx = track_x0 + (self._x + 2.4) / 4.8 * track_w
        cart_w = W * 0.12; cart_h = H * 0.08
        cart_rect = QRectF(cx - cart_w / 2, track_y - cart_h, cart_w, cart_h)

        color = MAGENTA if self._done else CYAN
        p.setPen(QPen(color, 2))
        p.setBrush(QColor(color.red(), color.green(), color.blue(), 40))
        p.drawRoundedRect(cart_rect, 4, 4)

        wr = cart_h * 0.4
        for wx in [cx - cart_w * 0.3, cx + cart_w * 0.3]:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 60))
            p.drawEllipse(QRectF(wx - wr, track_y - wr, 2 * wr, 2 * wr))

        pole_len = H * 0.35
        pole_px = cx + pole_len * math.sin(self._theta)
        pole_py = (track_y - cart_h) - pole_len * math.cos(self._theta)
        pivot = QPointF(cx, track_y - cart_h)
        tip = QPointF(pole_px, pole_py)

        pole_color = (EMERALD if abs(self._theta) < 0.1
                      else AMBER if abs(self._theta) < 0.15 else MAGENTA)
        g = QLinearGradient(pivot, tip)
        g.setColorAt(0, QColor(pole_color.red(), pole_color.green(), pole_color.blue(), 200))
        g.setColorAt(1, QColor(pole_color.red(), pole_color.green(), pole_color.blue(), 80))
        p.setPen(QPen(g, 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(pivot, tip)
        draw_glow(p, tip, 8, pole_color)

        labels = ["x", "ẋ", "θ", "θ̇"]
        vals = [self._x / 2.4, self._x_dot / 3.0,
                self._theta / 0.2094, self._theta_dot / 3.0]
        bar_h = H * 0.06; bar_y = H * 0.88
        bw = (W - 80) / 4; bpad = 8

        font = QFont(); font.setPointSize(8); p.setFont(font)
        for i, (lbl, v) in enumerate(zip(labels, vals)):
            bx = 40 + i * bw
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 15))
            p.drawRoundedRect(QRectF(bx, bar_y, bw - bpad, bar_h), 3, 3)
            fill_w = abs(v) * (bw - bpad) * 0.5
            fill_x = (bx + (bw - bpad) / 2 if v >= 0
                      else bx + (bw - bpad) / 2 - fill_w)
            bar_color = CYAN if v >= 0 else MAGENTA
            p.setBrush(QColor(bar_color.red(), bar_color.green(), bar_color.blue(), 160))
            p.drawRoundedRect(QRectF(fill_x, bar_y + 1, fill_w, bar_h - 2), 2, 2)
            p.setPen(WHITE_60)
            p.drawText(QRectF(bx, bar_y - 14, bw - bpad, 14),
                       Qt.AlignmentFlag.AlignCenter, lbl)

        p.end()
