from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                QLabel, QPushButton)

from ui.components.glass_panel import GlassPanel
from ui.components.learning_curve import LearningCurve
from ui.components.status_bar import StatusBar
from ui.visualizations.gridworld_view import GridWorldView
from controllers.arena_controller import ArenaController
from ui.theme import CYAN, VIOLET


class ArenaView(QWidget):
    """Side-by-side Q-Learning vs REINFORCE head-to-head."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctrl = ArenaController(self)
        self._ctrl.step_done.connect(self._on_step)
        self._ctrl.q_table_updated.connect(self._on_q)
        self._ctrl.rf_policy_updated.connect(self._on_rf)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16); lay.setSpacing(12)

        hdr = QLabel("Arena — Q-Learning vs REINFORCE")
        hdr.setStyleSheet("color:#FFFFFF;font-size:14pt;font-weight:900;")
        lay.addWidget(hdr)

        # Two GridWorld views
        env_row = QHBoxLayout(); env_row.setSpacing(16)

        ql_panel = GlassPanel(); ql_l = ql_panel.layout()
        ql_title = QLabel("Q-Learning")
        ql_title.setStyleSheet("color:#00D4FF;font-size:11pt;font-weight:700;")
        ql_l.addWidget(ql_title)
        self._ql_view = GridWorldView(); ql_l.addWidget(self._ql_view)
        env_row.addWidget(ql_panel)

        rf_panel = GlassPanel(); rf_l = rf_panel.layout()
        rf_title = QLabel("REINFORCE")
        rf_title.setStyleSheet("color:#7C3AED;font-size:11pt;font-weight:700;")
        rf_l.addWidget(rf_title)
        self._rf_view = GridWorldView(); rf_l.addWidget(self._rf_view)
        env_row.addWidget(rf_panel)

        lay.addLayout(env_row, 3)

        self._ql_view.load_env(self._ctrl._env_ql)
        self._rf_view.load_env(self._ctrl._env_rf)

        # Learning curves
        curve_row = QHBoxLayout(); curve_row.setSpacing(16)

        qcp = GlassPanel(); qcl = qcp.layout()
        qcl.addWidget(QLabel("Q-Learning Returns"))
        self._ql_curve = LearningCurve(line_color=CYAN); qcl.addWidget(self._ql_curve)
        curve_row.addWidget(qcp)

        rcp = GlassPanel(); rcl = rcp.layout()
        rcl.addWidget(QLabel("REINFORCE Returns"))
        self._rf_curve = LearningCurve(line_color=VIOLET); rcl.addWidget(self._rf_curve)
        curve_row.addWidget(rcp)

        lay.addLayout(curve_row, 2)

        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("▶  Start Arena")
        self._btn_stop  = QPushButton("■  Stop")
        self._btn_reset = QPushButton("↺  Reset")
        for btn in [self._btn_start, self._btn_stop, self._btn_reset]:
            btn.setFixedHeight(40); btn_row.addWidget(btn)
        lay.addLayout(btn_row)

        self._status = StatusBar(); lay.addWidget(self._status)

        self._btn_start.clicked.connect(self._ctrl.start)
        self._btn_stop.clicked.connect(self._ctrl.stop)
        self._btn_reset.clicked.connect(self._reset)

    def _on_step(self, ep: int, ql_r: float, rf_r: float):
        self._ql_curve.add_point(ql_r)
        self._rf_curve.add_point(rf_r)
        winner = "Q-Learning" if ql_r > rf_r else "REINFORCE" if rf_r > ql_r else "Tie"
        self._status.set_text(
            f"Ep {ep}  |  Q-Learning: {ql_r:.2f}  |  REINFORCE: {rf_r:.2f}  |  → {winner}")

    def _on_q(self, q_table):
        self._ql_view.update_q(q_table)

    def _on_rf(self, policy_w):
        self._rf_view.update_q(policy_w)

    def _reset(self):
        self._ctrl.reset()
        self._ql_curve.clear(); self._rf_curve.clear()
        self._ql_view.load_env(self._ctrl._env_ql)
        self._rf_view.load_env(self._ctrl._env_rf)
        self._status.set_text("Reset — press Start Arena")
