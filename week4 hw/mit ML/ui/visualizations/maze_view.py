from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QRectF, QPointF, Qt
from PySide6.QtGui import QPainter, QColor, QPen, QFont

from ui.theme import CYAN, MAGENTA, EMERALD, BG, WHITE_60
from ui.visualizations.painter_utils import draw_glow


class MazeView(QWidget):
    """Renders Maze walls, visited path trail, agent glow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self._walls: np.ndarray | None = None
        self._rows = 5; self._cols = 5
        self._agent_pos: tuple[int, int] = (0, 0)
        self._solution: list[tuple[int, int]] = []
        self._visited: list[tuple[int, int]] = []
        self._show_solution = False

    def load_maze(self, maze) -> None:
        self._walls = maze.walls.copy()
        self._rows = maze.rows; self._cols = maze.cols
        self._solution = maze.bfs_path()
        self._agent_pos = (0, 0)
        self._visited = [(0, 0)]
        self.update()

    def update_agent(self, pos: tuple[int, int]) -> None:
        self._agent_pos = pos
        if pos not in self._visited:
            self._visited.append(pos)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        if self._walls is None:
            p.end(); return

        W, H = self.width(), self.height()
        cs = min(W / self._cols, H / self._rows) * 0.9
        ox = (W - cs * self._cols) / 2
        oy = (H - cs * self._rows) / 2
        wall_w = max(2, cs * 0.08)

        def cell_rect(r, c):
            return QRectF(ox + c * cs, oy + r * cs, cs, cs)

        for r, c in self._visited:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 212, 255, 18))
            p.drawRect(cell_rect(r, c))

        if self._show_solution:
            for r, c in self._solution:
                p.setBrush(QColor(16, 185, 129, 35))
                p.drawRect(cell_rect(r, c))

        gr, gc = self._rows - 1, self._cols - 1
        goal_rect = cell_rect(gr, gc)
        p.setBrush(QColor(16, 185, 129, 60))
        p.drawRect(goal_rect)
        p.setPen(EMERALD)
        font = QFont(); font.setPointSize(int(cs * 0.3)); p.setFont(font)
        p.drawText(goal_rect, Qt.AlignmentFlag.AlignCenter, "★")

        p.setPen(QPen(QColor(255, 255, 255, 80), wall_w,
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
        for r in range(self._rows):
            for c in range(self._cols):
                x0 = ox + c * cs; y0 = oy + r * cs
                x1 = x0 + cs; y1 = y0 + cs
                walls = self._walls[r, c]  # [N, S, W, E]
                if walls[0]: p.drawLine(int(x0), int(y0), int(x1), int(y0))
                if walls[1]: p.drawLine(int(x0), int(y1), int(x1), int(y1))
                if walls[2]: p.drawLine(int(x0), int(y0), int(x0), int(y1))
                if walls[3]: p.drawLine(int(x1), int(y0), int(x1), int(y1))

        p.setPen(QPen(QColor(255, 255, 255, 120), wall_w))
        p.drawRect(QRectF(ox, oy, cs * self._cols, cs * self._rows))

        ar, ac = self._agent_pos
        ax = ox + ac * cs + cs / 2; ay = oy + ar * cs + cs / 2
        draw_glow(p, QPointF(ax, ay), cs * 0.3, CYAN)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(CYAN)
        r_dot = cs * 0.18
        p.drawEllipse(QRectF(ax - r_dot, ay - r_dot, 2 * r_dot, 2 * r_dot))

        p.end()
