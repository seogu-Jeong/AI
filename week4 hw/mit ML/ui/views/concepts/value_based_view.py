from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainter, QColor, QPen, QFont

from ui.theme import (CYAN, MAGENTA, VIOLET, EMERALD, AMBER,
                      BG, WHITE_60, WHITE_40, pulse, lerp_color)
from ui.visualizations.painter_utils import draw_glow, draw_glass_rect, heatmap_color


class QTableWidget(QWidget):
    def __init__(self, n_states: int = 16, n_actions: int = 4, parent=None):
        super().__init__(parent)
        self._n_states = min(n_states, 16)
        self._n_actions = n_actions
        self._q = np.zeros((self._n_states, self._n_actions))
        self.setMinimumHeight(160)

    def update_q(self, q: np.ndarray):
        self._q = q[:self._n_states].copy(); self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        nr = self._n_states; nc = self._n_actions
        col_w = (W - 60) / nc; row_h = (H - 30) / nr
        ox = 60; oy = 30

        headers = ["↑Up", "↓Dn", "←Lt", "→Rt"][:nc]
        f = QFont(); f.setPointSize(8); p.setFont(f)
        p.setPen(CYAN)
        for j, h in enumerate(headers):
            p.drawText(QRectF(ox + j*col_w, 4, col_w, 22),
                       Qt.AlignmentFlag.AlignCenter, h)

        min_q = float(np.min(self._q)); max_q = float(np.max(self._q))
        if max_q == min_q: max_q = min_q + 1

        for i in range(nr):
            p.setPen(WHITE_40)
            p.drawText(QRectF(0, oy+i*row_h, 56, row_h),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                       f"S{i}")
            for j in range(nc):
                val = self._q[i, j]
                cell = QRectF(ox+j*col_w+1, oy+i*row_h+1, col_w-2, row_h-2)
                c = heatmap_color(val, min_q, max_q); c.setAlpha(150)
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(c)
                p.drawRect(cell)
                p.setPen(WHITE_60)
                p.drawText(cell, Qt.AlignmentFlag.AlignCenter, f"{val:.2f}")
        p.end()


class BellmanWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self._phase = 0.0

    def set_phase(self, phase: float):
        self._phase = phase; self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        root = (W/2, H*0.2)
        children = [(W*0.22, H*0.72), (W*0.5, H*0.72), (W*0.78, H*0.72)]
        rewards = ["-0.01", "+1.0", "-10.0"]
        colors = [CYAN, EMERALD, MAGENTA]
        sel = int(self._phase * 3) % 3

        lf = QFont(); lf.setPointSize(8); p.setFont(lf)
        for i, (cx, cy) in enumerate(children):
            c = colors[i] if i == sel else WHITE_40
            p.setPen(QPen(c, 2 if i == sel else 1))
            p.drawLine(int(root[0]), int(root[1]+18), int(cx), int(cy-18))
            p.setPen(c)
            p.drawText(QRectF((root[0]+cx)/2-20, (root[1]+cy)/2-8, 40, 16),
                       Qt.AlignmentFlag.AlignCenter, f"r={rewards[i]}")
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 60 if i == sel else 20))
            p.setPen(QPen(c, 2))
            p.drawEllipse(QRectF(cx-18, cy-18, 36, 36))
            p.setPen(c)
            p.drawText(QRectF(cx-18, cy-18, 36, 36), Qt.AlignmentFlag.AlignCenter, "V(s')")

        br = pulse(self._phase, 0.7, 1.0)
        rc = QColor(0, int(212*br), int(255*br))
        p.setBrush(QColor(rc.red(), rc.green(), rc.blue(), 80))
        p.setPen(QPen(rc, 2))
        p.drawEllipse(QRectF(root[0]-24, root[1]-18, 48, 36))
        f2 = QFont(); f2.setPointSize(9); f2.setBold(True); p.setFont(f2)
        p.setPen(rc)
        p.drawText(QRectF(root[0]-24, root[1]-18, 48, 36),
                   Qt.AlignmentFlag.AlignCenter, "Q(s,a)")

        f3 = QFont(); f3.setPointSize(8); p.setFont(f3)
        p.setPen(AMBER)
        p.drawText(QRectF(W*0.05, H*0.88, W*0.9, 20), Qt.AlignmentFlag.AlignCenter,
                   "Q(s,a) = r + γ·max Q(s',a')  (Bellman equation)")
        p.end()


class EpsilonGreedyWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)
        self._epsilon = 0.2

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        cx = W/2; cy = H*0.5; r = min(W, H)*0.3
        eps = self._epsilon

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(MAGENTA.red(), MAGENTA.green(), MAGENTA.blue(), 180))
        p.drawPie(QRectF(cx-r, cy-r, 2*r, 2*r), 90*16, int(eps*360*16))
        p.setBrush(QColor(CYAN.red(), CYAN.green(), CYAN.blue(), 180))
        p.drawPie(QRectF(cx-r, cy-r, 2*r, 2*r),
                  int((90+eps*360)*16), int((1-eps)*360*16))

        f = QFont(); f.setPointSize(9); p.setFont(f)
        p.setPen(MAGENTA)
        p.drawText(QRectF(cx-r-90, cy-10, 80, 20),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"Explore ε={eps:.2f}")
        p.setPen(CYAN)
        p.drawText(QRectF(cx+r+10, cy-10, 100, 20),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"Exploit {1-eps:.2f}")
        p.end()


class ValueBasedView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20); lay.setSpacing(16)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        inner = QWidget(); il = QVBoxLayout(inner)
        il.setSpacing(20); il.setContentsMargins(0, 0, 0, 0)

        def section(t):
            lb = QLabel(t)
            lb.setStyleSheet("color:#00D4FF;font-size:12pt;font-weight:700;"
                             "border-bottom:1px solid rgba(0,212,255,0.25);padding-bottom:4px;")
            return lb

        il.addWidget(section("Q-Table (Live)"))
        self._qt = QTableWidget(n_states=16, n_actions=4); il.addWidget(self._qt)

        il.addWidget(section("Bellman Backup"))
        self._bellman = BellmanWidget(); il.addWidget(self._bellman)

        il.addWidget(section("ε-Greedy Exploration"))
        self._eps_w = EpsilonGreedyWidget(); il.addWidget(self._eps_w)

        il.addStretch()
        scroll.setWidget(inner); lay.addWidget(scroll)

    def set_phase(self, phase: float):
        self._bellman.set_phase(phase)

    def update_q(self, q: np.ndarray):
        self._qt.update_q(q)
