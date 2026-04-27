from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QRadialGradient

from ui.theme import CYAN, MAGENTA, EMERALD, AMBER, BG, SURFACE1, WHITE_60
from ui.visualizations.painter_utils import (
    draw_glow, draw_arrow, draw_glass_rect, heatmap_color
)


class FloatUp:
    """Animated reward float-up text."""
    def __init__(self, x: float, y: float, text: str, color: QColor):
        self.x = x; self.y = y; self.text = text; self.color = color
        self.life = 1.0

    def tick(self, dt: float = 0.05) -> bool:
        self.y -= 2.0; self.life -= dt
        return self.life > 0


class GridWorldView(QWidget):
    """Renders GridWorld Q-heatmap, policy arrows, agent glow, reward float-ups."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self._n = 4
        self._q_table: np.ndarray | None = None
        self._agent_state: int = 0
        self._obstacles: set[int] = set()
        self._cliffs: set[int] = set()
        self._goal: int = 0
        self._float_ups: list[FloatUp] = []
        self._phase = 0.0

        self._anim = QTimer(self)
        self._anim.timeout.connect(self._tick_floats)
        self._anim.start(33)

    def load_env(self, env) -> None:
        self._n = env.n
        self._obstacles = set(env.obstacles)
        self._cliffs = set(env.cliffs)
        self._goal = env.goal_state
        self._agent_state = env.agent_state
        self.update()

    def update_q(self, q_table: np.ndarray) -> None:
        self._q_table = q_table.copy(); self.update()

    def update_agent(self, state: int, reward: float) -> None:
        self._agent_state = state
        if abs(reward) > 0.05:
            cx, cy = self._state_center(state)
            color = EMERALD if reward > 0 else MAGENTA
            text = f"+{reward:.2f}" if reward > 0 else f"{reward:.2f}"
            self._float_ups.append(FloatUp(cx, cy, text, color))
        self.update()

    def set_phase(self, phase: float) -> None:
        self._phase = phase; self.update()

    def _tick_floats(self):
        self._float_ups = [f for f in self._float_ups if f.tick()]
        if self._float_ups: self.update()

    def _cell_size(self) -> float:
        return min(self.width(), self.height()) / self._n

    def _state_to_row_col(self, s: int) -> tuple[int, int]:
        return s // self._n, s % self._n

    def _state_center(self, s: int) -> tuple[float, float]:
        r, c = self._state_to_row_col(s)
        cs = self._cell_size()
        ox = (self.width() - cs * self._n) / 2
        oy = (self.height() - cs * self._n) / 2
        return ox + c * cs + cs / 2, oy + r * cs + cs / 2

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        cs = self._cell_size()
        ox = (self.width() - cs * self._n) / 2
        oy = (self.height() - cs * self._n) / 2

        for s in range(self._n * self._n):
            r, c = self._state_to_row_col(s)
            x = ox + c * cs; y = oy + r * cs
            rect = QRectF(x + 1, y + 1, cs - 2, cs - 2)

            if self._q_table is not None and s not in self._obstacles:
                max_q = float(np.max(self._q_table[s]))
                min_q = float(np.min(self._q_table))
                max_all = float(np.max(self._q_table))
                cell_color = heatmap_color(max_q, min_q, max_all)
                cell_color.setAlpha(100)
                p.fillRect(rect, cell_color)

            if s in self._obstacles:
                p.fillRect(rect, QColor(255, 255, 255, 20))
                p.setPen(QPen(QColor(255, 255, 255, 40), 1))
                p.drawLine(int(x + 2), int(y + 2), int(x + cs - 2), int(y + cs - 2))
                p.drawLine(int(x + cs - 2), int(y + 2), int(x + 2), int(y + cs - 2))
            elif s in self._cliffs:
                p.fillRect(rect, QColor(255, 0, 110, 60))
            elif s == self._goal:
                p.fillRect(rect, QColor(16, 185, 129, 80))
                p.setPen(QPen(EMERALD, 1))
                font = QFont(); font.setPointSize(int(cs * 0.3)); p.setFont(font)
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "★")

            p.setPen(QPen(QColor(255, 255, 255, 15), 1))
            p.drawRect(rect)

            if (self._q_table is not None and s not in self._obstacles
                    and s != self._goal and s not in self._cliffs):
                best_a = int(np.argmax(self._q_table[s]))
                cx_c = x + cs / 2; cy_c = y + cs / 2
                draw_arrow(p, cx_c, cy_c, best_a, cs * 0.28, WHITE_60, 0.6)

        ax, ay = self._state_center(self._agent_state)
        draw_glow(p, QPointF(ax, ay), cs * 0.35, CYAN)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(CYAN)
        r = cs * 0.18
        p.drawEllipse(QRectF(ax - r, ay - r, 2 * r, 2 * r))

        font = QFont(); font.setPointSize(9); font.setBold(True); p.setFont(font)
        for fu in self._float_ups:
            alpha = int(fu.life * 220)
            col = QColor(fu.color); col.setAlpha(alpha)
            p.setPen(col)
            p.drawText(QRectF(fu.x - 30, fu.y - 10, 60, 20),
                       Qt.AlignmentFlag.AlignCenter, fu.text)
        p.end()
