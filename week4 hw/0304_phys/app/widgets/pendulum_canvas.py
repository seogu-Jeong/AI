"""
PendulumCanvas — QPainter-based pendulum animation widget.
PRD-04 §3.2: pivot, rod, bob (16px, #2196F3), trail (30 pts), angle arc+label.
No Matplotlib. Driven by PendulumAnimationController via set_state().
"""
import math
from collections import deque
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer
from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath, QFont


class PendulumCanvas(QWidget):
    """QPainter pendulum animation. Call set_state(theta_rad) each frame."""

    BOB_COLOR  = QColor('#2196F3')
    TRAIL_LEN  = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._theta_rad = 0.0
        self._L_norm    = 1.0   # normalized 0..1 for scaling
        self._trail: deque = deque(maxlen=self.TRAIL_LEN)
        self._dark = False
        self._theta2_rad: float = None   # second pendulum (isochronism demo)
        self._trail2: deque = deque(maxlen=self.TRAIL_LEN)

    def set_state(self, theta_rad: float):
        """Primary pendulum angle. Triggers repaint."""
        self._theta_rad = theta_rad
        pivot, bob = self._compute_positions(theta_rad)
        self._trail.append(QPointF(*bob))
        self.update()

    def set_secondary_state(self, theta_rad: float):
        """Second pendulum (isochronism demo, orange)."""
        self._theta2_rad = theta_rad
        _, bob = self._compute_positions(theta_rad)
        self._trail2.append(QPointF(*bob))

    def clear_secondary(self):
        self._theta2_rad = None
        self._trail2.clear()

    def set_pendulum_length(self, L: float, L_max: float = 3.0):
        self._L_norm = min(L / L_max, 1.0)
        self._trail.clear(); self._trail2.clear()

    def set_dark(self, dark: bool):
        self._dark = dark; self.update()

    def _rod_px(self) -> float:
        return min(self.width(), self.height()) * 0.35 * (0.5 + 0.5 * self._L_norm)

    def _pivot(self):
        return (self.width() / 2, self.height() * 0.18)

    def _compute_positions(self, theta_rad: float):
        px, py = self._pivot()
        rod = self._rod_px()
        bx = px + rod * math.sin(theta_rad)
        by = py + rod * math.cos(theta_rad)
        return (px, py), (bx, by)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg = QColor('#1E1E2E') if self._dark else QColor('#FFFFFF')
        p.fillRect(self.rect(), bg)
        fg_color = QColor('#CDD6F4') if self._dark else QColor('#212121')

        pivot, bob = self._compute_positions(self._theta_rad)

        # Trail (fading alpha)
        trail_list = list(self._trail)
        for i, pt in enumerate(trail_list):
            alpha = int(255 * (i + 1) / len(trail_list)) if trail_list else 255
            trail_color = QColor(self.BOB_COLOR)
            trail_color.setAlpha(alpha // 3)
            p.setPen(QPen(trail_color, 1.5))
            if i > 0:
                p.drawLine(trail_list[i-1], pt)

        # Second pendulum trail (orange)
        if self._theta2_rad is not None:
            trail2 = list(self._trail2)
            orange = QColor('#FF9800')
            for i, pt in enumerate(trail2):
                alpha = int(255 * (i + 1) / len(trail2)) if trail2 else 255
                tc = QColor(orange); tc.setAlpha(alpha // 3)
                p.setPen(QPen(tc, 1.5))
                if i > 0: p.drawLine(trail2[i-1], pt)

        # Rod
        p.setPen(QPen(fg_color, 2))
        p.drawLine(QPointF(*pivot), QPointF(*bob))

        # Second rod
        if self._theta2_rad is not None:
            _, bob2 = self._compute_positions(self._theta2_rad)
            p.setPen(QPen(QColor('#FF9800'), 2))
            p.drawLine(QPointF(*pivot), QPointF(*bob2))
            p.setBrush(QBrush(QColor('#FF9800')))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(*bob2), 12, 12)

        # Pivot
        p.setBrush(QBrush(fg_color)); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(*pivot), 6, 6)

        # Bob
        p.setBrush(QBrush(self.BOB_COLOR)); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(*bob), 16, 16)

        # Angle arc
        px, py = pivot
        arc_r = 40.0
        p.setPen(QPen(fg_color, 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        rect = QRectF(px - arc_r, py - arc_r, arc_r*2, arc_r*2)
        start_angle = 90 * 16   # 12-o'clock in Qt units (16ths of degree)
        span_angle  = int(-math.degrees(self._theta_rad) * 16)
        if abs(span_angle) > 16:
            p.drawArc(rect, start_angle, span_angle)

        # Angle label
        p.setPen(fg_color)
        f = QFont(); f.setPointSize(10); p.setFont(f)
        deg = math.degrees(self._theta_rad)
        p.drawText(int(px + 48), int(py + 16), f"θ = {deg:.1f}°")

        p.end()


class PendulumAnimationController:
    """Drives PendulumCanvas at TARGET_FPS using QTimer. SDD-PHYSAI-004 §4."""
    TARGET_FPS    = 30
    BASE_INTERVAL = 1000 // TARGET_FPS   # 33 ms

    def __init__(self, canvas: PendulumCanvas):
        self._canvas     = canvas
        self._timer      = QTimer()
        self._timer.timeout.connect(self._advance)
        self._traj       = None   # shape (N, 2): [theta_rad, omega]
        self._traj2      = None   # optional second pendulum
        self._frame      = 0
        self._speed      = 1.0

    def load(self, traj, L: float, traj2=None):
        """Load primary (and optional secondary) trajectory, reset to frame 0."""
        self._traj  = traj
        self._traj2 = traj2
        self._frame = 0
        self._canvas.set_pendulum_length(L)
        if traj2 is None: self._canvas.clear_secondary()

    def play(self):
        interval = max(1, int(self.BASE_INTERVAL / self._speed))
        self._timer.start(interval)

    def pause(self): self._timer.stop()

    def toggle(self):
        self.pause() if self._timer.isActive() else self.play()

    def set_speed(self, multiplier: float):
        self._speed = multiplier
        if self._timer.isActive():
            self._timer.setInterval(max(1, int(self.BASE_INTERVAL / multiplier)))

    def is_playing(self) -> bool:
        return self._timer.isActive()

    def _advance(self):
        if self._traj is None: return
        N = len(self._traj)
        if self._frame >= N: self._frame = 0
        self._canvas.set_state(float(self._traj[self._frame, 0]))
        if self._traj2 is not None and self._frame < len(self._traj2):
            self._canvas.set_secondary_state(float(self._traj2[self._frame, 0]))
        self._frame += 1
