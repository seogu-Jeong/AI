from __future__ import annotations
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainter, QColor, QFont

from ui.theme import CYAN, MAGENTA, VIOLET, EMERALD, AMBER, BG, WHITE_60
from ui.visualizations.painter_utils import draw_glass_rect


class ApplicationCard(QWidget):
    def __init__(self, title, year, desc, result, color, parent=None):
        super().__init__(parent)
        self._title = title; self._year = year
        self._desc = desc; self._result = result; self._color = color
        self.setMinimumHeight(115)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        draw_glass_rect(p, QRectF(4, 4, self.width()-8, self.height()-8))

        W, H = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(self._color)
        p.drawRoundedRect(QRectF(4, 4, 4, H-8), 2, 2)

        f = QFont(); f.setPointSize(11); f.setBold(True); p.setFont(f)
        p.setPen(self._color)
        p.drawText(QRectF(20, 8, W-100, 28), Qt.AlignmentFlag.AlignVCenter, self._title)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self._color.red(), self._color.green(), self._color.blue(), 40))
        p.drawRoundedRect(QRectF(W-70, 8, 60, 22), 4, 4)
        f2 = QFont(); f2.setPointSize(8); p.setFont(f2); p.setPen(self._color)
        p.drawText(QRectF(W-70, 8, 60, 22), Qt.AlignmentFlag.AlignCenter, self._year)

        f3 = QFont(); f3.setPointSize(9); p.setFont(f3); p.setPen(WHITE_60)
        p.drawText(QRectF(20, 40, W-30, H-76), Qt.AlignmentFlag.AlignTop, self._desc)

        f4 = QFont(); f4.setPointSize(8); f4.setBold(True); p.setFont(f4)
        p.setPen(self._color)
        p.drawText(QRectF(20, H-26, W-30, 22),
                   Qt.AlignmentFlag.AlignVCenter, f"★  {self._result}")
        p.end()


APPLICATIONS = [
    ("DQN — Atari", "2013",
     "DeepMind trains CNN+Q-Learning on raw Atari pixels. 49 games, superhuman on 29.",
     "Human-level control from pixels", CYAN),
    ("AlphaGo / AlphaZero", "2016–17",
     "Combines MCTS with policy+value networks. Self-play from scratch.",
     "Defeated world champion Lee Sedol (4–1)", VIOLET),
    ("OpenAI Five — Dota 2", "2019",
     "5-agent team, multi-agent RL, 180 years of self-play/day.",
     "World championship defeat — OG team", EMERALD),
    ("AlphaStar — StarCraft II", "2019",
     "Multiagent population-based RL, 200 years training, 500 APM.",
     "Grandmaster level in all three races", AMBER),
    ("Robotics — Dexterous Manipulation", "2019",
     "OpenAI Dactyl: domain randomization + PPO on Shadow Hand.",
     "Solved Rubik's cube one-handed", MAGENTA),
    ("ChatGPT — RLHF", "2022",
     "Reinforcement Learning from Human Feedback fine-tunes LLMs.\n"
     "Reward model from human preference labels.",
     "State-of-the-art conversational AI", QColor(255, 200, 50)),
]


class ApplicationsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20); lay.setSpacing(12)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        inner = QWidget(); il = QVBoxLayout(inner)
        il.setSpacing(12); il.setContentsMargins(0, 0, 0, 0)

        hdr = QLabel("Real-World RL Applications")
        hdr.setStyleSheet("color:#10B981;font-size:13pt;font-weight:900;"
                          "border-bottom:1px solid rgba(16,185,129,0.3);padding-bottom:6px;")
        il.addWidget(hdr)

        for args in APPLICATIONS:
            il.addWidget(ApplicationCard(*args))

        il.addStretch()
        scroll.setWidget(inner); lay.addWidget(scroll)
