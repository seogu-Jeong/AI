"""
PendulumCanvas — QPainter-based real-time pendulum animation widget.
Receives angular state via set_state(); does not own the animation timer.
"""
import math
from collections import deque

from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont,
    QRadialGradient, QLinearGradient, QPaintEvent,
)
from PySide6.QtWidgets import QWidget, QSizePolicy


class PendulumCanvas(QWidget):
    """
    Renders a pendulum with:
    - Gradient bob with specular highlight
    - Motion trail (fading alpha)
    - Angle arc indicator
    - Energy bar (KE / PE)
    """
    TRAIL_LEN  = 40
    BOB_R      = 18
    PIVOT_R    = 7

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._theta  : float        = 0.0
        self._omega  : float        = 0.0
        self._L      : float        = 1.0
        self._trail  : deque        = deque(maxlen=self.TRAIL_LEN)
        self._dark   : bool         = False

        # Energy display
        self._E0     : float = 0.0
        self._KE     : float = 0.0
        self._PE     : float = 0.0

    # ── public API ───────────────────────────────────────────────────────────
    def set_state(self, theta_rad: float, omega_rad_s: float = 0.0):
        self._theta = theta_rad
        self._omega = omega_rad_s
        px, py = self._bob_px()
        self._trail.append((px, py))
        g = 9.81
        self._KE = 0.5 * self._L ** 2 * omega_rad_s ** 2
        self._PE = g * self._L * (1 - math.cos(theta_rad))
        self.update()

    def set_pendulum_length(self, L: float):
        self._L = L
        self._trail.clear()
        self.update()

    def set_dark(self, dark: bool):
        self._dark = dark
        self.update()

    def set_energy_reference(self, E0: float):
        self._E0 = E0

    # ── painting ─────────────────────────────────────────────────────────────
    def paintEvent(self, event: QPaintEvent):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg = QColor('#181825') if self._dark else QColor('#FFFFFF')
        p.fillRect(self.rect(), bg)

        pivot = self._pivot()
        bob   = QPoint(*self._bob_px())

        self._draw_ceiling(p, pivot)
        self._draw_trail(p)
        self._draw_angle_arc(p, pivot)
        self._draw_rod(p, pivot, bob)
        self._draw_bob(p, bob)
        self._draw_pivot(p, pivot)
        self._draw_hud(p, pivot)

    # ── draw sub-routines ────────────────────────────────────────────────────
    def _draw_ceiling(self, p: QPainter, pivot: QPoint):
        color = QColor('#313244') if self._dark else QColor('#E0E0E0')
        pen = QPen(color, 3)
        p.setPen(pen)
        p.drawLine(0, pivot.y(), self.width(), pivot.y())
        # hatching marks
        pen.setWidth(1)
        p.setPen(pen)
        for x in range(0, self.width(), 12):
            p.drawLine(x, pivot.y(), x - 8, pivot.y() - 8)

    def _draw_trail(self, p: QPainter):
        trail = list(self._trail)
        n = len(trail)
        for i in range(1, n):
            alpha = int(20 + 220 * (i / n) ** 1.5)
            blue  = QColor(137, 180, 250, alpha) if self._dark else QColor(25, 118, 210, alpha)
            p.setPen(QPen(blue, max(1, 2 * i // n)))
            p.drawLine(int(trail[i-1][0]), int(trail[i-1][1]),
                       int(trail[i][0]),   int(trail[i][1]))

    def _draw_angle_arc(self, p: QPainter, pivot: QPoint):
        if abs(self._theta) < 0.02:
            return
        r = 50
        color = QColor('#F38BA8') if self._dark else QColor('#E53935')
        pen = QPen(color, 1, Qt.PenStyle.DashLine)
        p.setPen(pen)
        # vertical reference
        p.drawLine(pivot.x(), pivot.y(), pivot.x(), pivot.y() + r + 15)
        # arc
        start_angle = 90 * 16
        span_angle  = int(-math.degrees(self._theta) * 16)
        p.drawArc(pivot.x() - r, pivot.y(), r * 2, r * 2, start_angle, span_angle)

    def _draw_rod(self, p: QPainter, pivot: QPoint, bob: QPoint):
        rod_col = QColor('#585B70') if self._dark else QColor('#455A64')
        p.setPen(QPen(rod_col, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(pivot, bob)

    def _draw_bob(self, p: QPainter, center: QPoint):
        r = self.BOB_R
        # Radial gradient for 3-D sphere effect
        grad = QRadialGradient(center.x() - r//3, center.y() - r//3, r * 2)
        if self._dark:
            grad.setColorAt(0.0, QColor('#74C7EC'))
            grad.setColorAt(0.6, QColor('#89B4FA'))
            grad.setColorAt(1.0, QColor('#1E66F5'))
        else:
            grad.setColorAt(0.0, QColor('#90CAF9'))
            grad.setColorAt(0.6, QColor('#1976D2'))
            grad.setColorAt(1.0, QColor('#0D47A1'))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(center, r, r)
        # specular
        p.setBrush(QBrush(QColor(255, 255, 255, 90)))
        p.drawEllipse(center - QPoint(r // 3, r // 3), r // 4, r // 4)

    def _draw_pivot(self, p: QPainter, center: QPoint):
        r = self.PIVOT_R
        color = QColor('#CDD6F4') if self._dark else QColor('#37474F')
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        p.drawEllipse(center, r, r)

    def _draw_hud(self, p: QPainter, pivot: QPoint):
        fg    = QColor('#CDD6F4') if self._dark else QColor('#455A64')
        small = QFont('Courier New', 9)
        p.setFont(small)
        p.setPen(fg)

        deg = math.degrees(self._theta)
        lines = [
            f"θ = {deg:+.2f}°",
            f"ω = {self._omega:+.3f} rad/s",
            f"L = {self._L:.2f} m",
        ]
        x0 = 10
        y0 = self.height() - 20 - len(lines) * 16
        for i, line in enumerate(lines):
            p.drawText(x0, y0 + i * 16, line)

        # Energy bar
        if self._E0 > 0:
            bar_w, bar_h = 80, 8
            bx = self.width() - bar_w - 10
            by = self.height() - 50
            E_total = self._KE + self._PE
            ke_frac = min(1.0, self._KE / max(E_total, 1e-9))
            pe_frac = 1.0 - ke_frac

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor('#313244') if self._dark else QColor('#E0E0E0'))
            p.drawRoundedRect(bx, by, bar_w, bar_h, 3, 3)
            p.setBrush(QColor('#F38BA8') if self._dark else QColor('#E53935'))
            p.drawRoundedRect(bx, by, int(bar_w * pe_frac), bar_h, 3, 3)
            p.setBrush(QColor('#89B4FA') if self._dark else QColor('#1976D2'))
            p.drawRoundedRect(bx + int(bar_w * pe_frac), by,
                              int(bar_w * ke_frac), bar_h, 3, 3)

            small2 = QFont('Courier New', 8)
            p.setFont(small2)
            p.setPen(fg)
            p.drawText(bx, by - 4, "PE    KE")

    # ── geometry helpers ─────────────────────────────────────────────────────
    def _pivot(self) -> QPoint:
        return QPoint(self.width() // 2, 55)

    def _rod_len(self) -> float:
        L_max   = 3.0
        max_rod = min(self.width() // 2 - 20, self.height() - 160)
        return max(30.0, max_rod * (self._L / L_max))

    def _bob_px(self) -> tuple:
        pivot = self._pivot()
        rod   = self._rod_len()
        px = pivot.x() + rod * math.sin(self._theta)
        py = pivot.y() + rod * math.cos(self._theta)
        return int(px), int(py)
