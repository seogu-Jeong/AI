from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import QRectF, QPointF, Qt
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient, QPolygonF

from ui.theme import (CYAN, MAGENTA, VIOLET, EMERALD, AMBER,
                      BG, WHITE_60, WHITE_40, pulse)
from ui.visualizations.painter_utils import draw_glow, draw_glass_rect


class PolicyWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self._probs = np.array([0.1, 0.5, 0.3, 0.1])

    def set_probs(self, probs: np.ndarray):
        self._probs = np.clip(probs, 0, 1); self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        n = len(self._probs); bw = (W-60)/n; bpad = 8
        max_h = H*0.58; oy = H*0.12

        f = QFont(); f.setPointSize(9); p.setFont(f)
        labels = ["↑", "↓", "←", "→"][:n]
        for i, (prob, lbl) in enumerate(zip(self._probs, labels)):
            bx = 30 + i*bw; bh = prob*max_h; by = oy+max_h-bh
            color = VIOLET if i == int(np.argmax(self._probs)) else CYAN
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(color.red(), color.green(), color.blue(), 160))
            p.drawRoundedRect(QRectF(bx+bpad/2, by, bw-bpad, bh), 4, 4)
            p.setPen(WHITE_60)
            p.drawText(QRectF(bx, oy+max_h+4, bw, 18), Qt.AlignmentFlag.AlignCenter, lbl)
            p.drawText(QRectF(bx, by-16, bw, 14), Qt.AlignmentFlag.AlignCenter,
                       f"{prob:.2f}")

        f2 = QFont(); f2.setPointSize(8); p.setFont(f2)
        p.setPen(WHITE_60)
        p.drawText(QRectF(0, H-18, W, 16), Qt.AlignmentFlag.AlignCenter,
                   "π(a|s) — softmax policy probabilities")
        p.end()


class PolicyGradientWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self._phase = 0.0
        self._history: list[float] = []

    def set_phase(self, phase: float):
        self._phase = phase; self.update()

    def add_reward(self, r: float):
        self._history.append(r)
        if len(self._history) > 60: self._history.pop(0)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        if len(self._history) < 2:
            f = QFont(); f.setPointSize(9); p.setFont(f); p.setPen(WHITE_40)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Run Arena → REINFORCE to see gradient ascent")
            p.end(); return

        n = len(self._history)
        cx0 = 40; cx1 = W-20; cy0 = H*0.1; cy1 = H*0.85
        cw = cx1-cx0; ch = cy1-cy0
        min_r = min(self._history); max_r = max(self._history)
        span = max(max_r-min_r, 1.0)

        def sc(i, v):
            return cx0 + i/max(n-1,1)*cw, cy1 - (v-min_r)/span*ch

        g = QLinearGradient(0, cy0, 0, cy1)
        g.setColorAt(0, QColor(124, 58, 237, 60))
        g.setColorAt(1, QColor(124, 58, 237, 0))
        poly = [QPointF(cx0, cy1)]
        for i, v in enumerate(self._history):
            sx, sy = sc(i, v); poly.append(QPointF(sx, sy))
        poly.append(QPointF(cx1, cy1))
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(g)
        p.drawPolygon(QPolygonF(poly))

        p.setPen(QPen(VIOLET, 2))
        for i in range(1, n):
            x1, y1 = sc(i-1, self._history[i-1])
            x2, y2 = sc(i, self._history[i])
            p.drawLine(int(x1), int(y1), int(x2), int(y2))

        lx, ly = sc(n-1, self._history[-1])
        draw_glow(p, QPointF(lx, ly), 6, VIOLET)

        f = QFont(); f.setPointSize(8); p.setFont(f); p.setPen(WHITE_60)
        p.drawText(QRectF(0, H-18, W, 16), Qt.AlignmentFlag.AlignCenter,
                   "REINFORCE — Episode Returns (Policy Gradient)")
        p.end()


class PolicyBasedView(QWidget):
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
            lb.setStyleSheet("color:#7C3AED;font-size:12pt;font-weight:700;"
                             "border-bottom:1px solid rgba(124,58,237,0.35);padding-bottom:4px;")
            return lb

        il.addWidget(section("Stochastic Policy π(a|s)"))
        self._pol = PolicyWidget(); il.addWidget(self._pol)

        il.addWidget(section("Policy Gradient — Episode Returns"))
        self._pg = PolicyGradientWidget(); il.addWidget(self._pg)

        il.addWidget(section("REINFORCE Algorithm"))
        txt = QLabel(
            "1. Run episode under current policy π_θ\n"
            "2. Compute discounted returns  Gₜ = Σ γᵏ rₜ₊ₖ\n"
            "3. Normalize: Ĝₜ = (Gₜ − μ) / (σ + ε)\n"
            "4. Update θ ← θ + α Σₜ Ĝₜ ∇θ log π_θ(aₜ|sₜ)\n"
            "5. Repeat until convergence"
        )
        txt.setStyleSheet(
            "color:rgba(255,255,255,0.7);font-family:monospace;font-size:10pt;"
            "padding:12px;background:rgba(255,255,255,0.05);border-radius:8px;")
        il.addWidget(txt)

        il.addStretch()
        scroll.setWidget(inner); lay.addWidget(scroll)

    def set_phase(self, phase: float):
        self._pg.set_phase(phase)

    def update_probs(self, probs: np.ndarray):
        self._pol.set_probs(probs)

    def add_reward(self, r: float):
        self._pg.add_reward(r)
