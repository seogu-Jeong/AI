from __future__ import annotations
import math
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import QRectF, QPointF, Qt, Signal
from PySide6.QtGui import (QPainter, QColor, QPen, QFont, QLinearGradient,
                            QPolygonF)

from ui.theme import (CYAN, MAGENTA, VIOLET, EMERALD, AMBER,
                      BG, SURFACE1, WHITE_60, WHITE_40, pulse, lerp_color)
from ui.visualizations.painter_utils import draw_glow, draw_glass_rect


class AgentEnvLoopWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self._phase = 0.0

    def set_phase(self, phase: float):
        self._phase = phase; self.update()

    def _arrowhead(self, p: QPainter, tip_x, tip_y, right: bool):
        d = 8
        if right:
            pts = [QPointF(tip_x, tip_y),
                   QPointF(tip_x - d, tip_y - d / 2),
                   QPointF(tip_x - d, tip_y + d / 2)]
        else:
            pts = [QPointF(tip_x, tip_y),
                   QPointF(tip_x + d, tip_y - d / 2),
                   QPointF(tip_x + d, tip_y + d / 2)]
        p.drawPolygon(QPolygonF(pts))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        bw = W * 0.26; bh = H * 0.38
        ag_x = W * 0.10; env_x = W * 0.60
        ag_y = (H - bh) / 2; env_y = ag_y

        draw_glass_rect(p, QRectF(ag_x, ag_y, bw, bh))
        draw_glass_rect(p, QRectF(env_x, env_y, bw, bh))

        font = QFont(); font.setPointSize(11); font.setBold(True); p.setFont(font)
        p.setPen(CYAN)
        p.drawText(QRectF(ag_x, ag_y, bw, bh), Qt.AlignmentFlag.AlignCenter, "Agent")
        p.setPen(EMERALD)
        p.drawText(QRectF(env_x, env_y, bw, bh), Qt.AlignmentFlag.AlignCenter,
                   "Environ\nment")

        cx_ag_r = ag_x + bw; cx_env_l = env_x

        # Action arrow (top, cyan)
        p.setPen(QPen(CYAN, 2))
        p.drawLine(int(cx_ag_r), int(ag_y + bh * 0.3),
                   int(cx_env_l), int(env_y + bh * 0.3))
        p.setBrush(CYAN); p.setPen(Qt.PenStyle.NoPen)
        self._arrowhead(p, cx_env_l, env_y + bh * 0.3, right=True)

        lf = QFont(); lf.setPointSize(8); p.setFont(lf)
        mid_x = (cx_ag_r + cx_env_l) / 2
        p.setPen(CYAN)
        p.drawText(QRectF(cx_ag_r, ag_y + bh * 0.3 - 18, mid_x - cx_ag_r, 16),
                   Qt.AlignmentFlag.AlignCenter, "Action aₜ")

        t_a = (self._phase * 1.5) % 1.0
        dot_x = cx_ag_r + t_a * (cx_env_l - cx_ag_r)
        draw_glow(p, QPointF(dot_x, ag_y + bh * 0.3), 5, CYAN)

        # State + Reward arrow (bottom, magenta)
        p.setPen(QPen(MAGENTA, 2))
        p.drawLine(int(cx_env_l), int(env_y + bh * 0.7),
                   int(cx_ag_r), int(ag_y + bh * 0.7))
        p.setBrush(MAGENTA); p.setPen(Qt.PenStyle.NoPen)
        self._arrowhead(p, cx_ag_r, ag_y + bh * 0.7, right=False)

        p.setPen(MAGENTA); p.setFont(lf)
        p.drawText(QRectF(cx_ag_r, ag_y + bh * 0.7 + 2, mid_x - cx_ag_r, 16),
                   Qt.AlignmentFlag.AlignCenter, "State sₜ₊₁, Reward rₜ")

        t_s = ((self._phase * 1.5) + 0.5) % 1.0
        dot_x2 = cx_env_l + t_s * (cx_ag_r - cx_env_l)
        draw_glow(p, QPointF(dot_x2, ag_y + bh * 0.7), 5, MAGENTA)

        p.end()


class MDPWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self._phase = 0.0

    def set_phase(self, phase: float):
        self._phase = phase; self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        centers = [QPointF(W*0.12, H*0.5), QPointF(W*0.38, H*0.28),
                   QPointF(W*0.63, H*0.5), QPointF(W*0.88, H*0.5)]
        labels = ["S₀", "S₁", "S₂", "S₃(G)"]
        colors = [CYAN, VIOLET, AMBER, EMERALD]
        r_s = 20

        transitions = [(0, 1, "a=0\nr=−0.01"), (0, 2, "a=1\nr=−0.01"),
                       (1, 2, "a=1\nr=−0.01"), (2, 3, "a=1\nr=+1.0")]

        lf = QFont(); lf.setPointSize(7); p.setFont(lf)
        p.setPen(QPen(WHITE_40, 1.5))
        for s_from, s_to, lbl in transitions:
            c1 = centers[s_from]; c2 = centers[s_to]
            p.drawLine(c1, c2)
            mid = QPointF((c1.x()+c2.x())/2, (c1.y()+c2.y())/2 - 10)
            p.setPen(WHITE_40)
            p.drawText(QRectF(mid.x()-25, mid.y()-10, 50, 20),
                       Qt.AlignmentFlag.AlignCenter, lbl)

        n = len(transitions); t = self._phase % 1.0
        seg = int(t * n); frac = (t * n) % 1.0
        sf, st, _ = transitions[min(seg, n-1)]
        c1 = centers[sf]; c2 = centers[st]
        dot = QPointF(c1.x() + frac*(c2.x()-c1.x()), c1.y() + frac*(c2.y()-c1.y()))
        draw_glow(p, dot, 6, CYAN)

        f2 = QFont(); f2.setPointSize(9); f2.setBold(True)
        for c, lbl, col in zip(centers, labels, colors):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(col.red(), col.green(), col.blue(), 40))
            p.drawEllipse(QRectF(c.x()-r_s, c.y()-r_s, 2*r_s, 2*r_s))
            p.setPen(QPen(col, 2)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(c.x()-r_s, c.y()-r_s, 2*r_s, 2*r_s))
            p.setPen(col); p.setFont(f2)
            p.drawText(QRectF(c.x()-r_s, c.y()-r_s, 2*r_s, 2*r_s),
                       Qt.AlignmentFlag.AlignCenter, lbl)
        p.end()


class DiscountWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self._gamma = 0.9

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        n = 8; bar_w = (W - 80) / n; bar_max_h = H * 0.58
        f = QFont(); f.setPointSize(8); p.setFont(f)
        for i in range(n):
            val = self._gamma ** i
            bh = val * bar_max_h; bx = 40 + i * bar_w; by = H * 0.72 - bh
            color = lerp_color(CYAN, MAGENTA, i / (n - 1))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(color.red(), color.green(), color.blue(), 160))
            p.drawRoundedRect(QRectF(bx+2, by, bar_w-4, bh), 3, 3)
            p.setPen(WHITE_60)
            p.drawText(QRectF(bx, H*0.75, bar_w, 20), Qt.AlignmentFlag.AlignCenter,
                       f"γ{i}")
            p.drawText(QRectF(bx, by-18, bar_w, 16), Qt.AlignmentFlag.AlignCenter,
                       f"{val:.2f}")
        f2 = QFont(); f2.setPointSize(9); f2.setBold(True); p.setFont(f2)
        p.setPen(WHITE_60)
        p.drawText(QRectF(0, 4, W, 20), Qt.AlignmentFlag.AlignCenter,
                   f"Discount Factor γ = {self._gamma:.2f}")
        p.end()


class RLBasicsView(QWidget):
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

        il.addWidget(section("Agent–Environment Loop"))
        self._loop = AgentEnvLoopWidget(); il.addWidget(self._loop)

        il.addWidget(section("Markov Decision Process (MDP)"))
        self._mdp = MDPWidget(); il.addWidget(self._mdp)

        il.addWidget(section("Discount Factor"))
        self._disc = DiscountWidget(); il.addWidget(self._disc)

        il.addStretch()
        scroll.setWidget(inner); lay.addWidget(scroll)

    def set_phase(self, phase: float):
        self._loop.set_phase(phase)
        self._mdp.set_phase(phase)
