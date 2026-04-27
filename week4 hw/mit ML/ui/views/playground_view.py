from __future__ import annotations
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                QTabWidget, QLabel, QPushButton)
from PySide6.QtCore import Qt

from ui.components.glass_panel import GlassPanel
from ui.components.slider_group import SliderGroup
from ui.components.learning_curve import LearningCurve
from ui.components.status_bar import StatusBar
from ui.visualizations.gridworld_view import GridWorldView
from ui.visualizations.cartpole_view import CartPoleView
from ui.visualizations.maze_view import MazeView
from controllers.training_controller import TrainingController


class GridWorldTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctrl = TrainingController(self)
        self._ctrl.agent_moved.connect(self._on_agent)
        self._ctrl.episode_done.connect(self._on_episode)

        lay = QHBoxLayout(self); lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(12)

        left = QVBoxLayout()
        self._gw_view = GridWorldView(); left.addWidget(self._gw_view)
        self._status = StatusBar(); left.addWidget(self._status)
        lay.addLayout(left, 3)

        right = QVBoxLayout(); right.setSpacing(10)
        self._sliders = SliderGroup([
            ("Grid Size",  4,  10,  4),
            ("Alpha α",    1,  20,  2),
            ("Gamma γ",   50,  99, 95),
            ("Epsilon ε",  5,  50, 20),
            ("Speed",      1,  50, 10),
        ])
        panel = GlassPanel(); pl = panel.layout()
        pl.addWidget(QLabel("Hyperparameters")); pl.addWidget(self._sliders)
        right.addWidget(panel)

        self._curve = LearningCurve()
        cp = GlassPanel(); cl = cp.layout()
        cl.addWidget(QLabel("Episode Rewards")); cl.addWidget(self._curve)
        right.addWidget(cp)

        btn_lay = QHBoxLayout()
        self._btn_start = QPushButton("▶  Train")
        self._btn_stop  = QPushButton("■  Stop")
        self._btn_reset = QPushButton("↺  Reset")
        for btn in [self._btn_start, self._btn_stop, self._btn_reset]:
            btn.setFixedHeight(36); btn_lay.addWidget(btn)
        right.addLayout(btn_lay); right.addStretch()
        lay.addLayout(right, 2)

        self._btn_start.clicked.connect(self._start)
        self._btn_stop.clicked.connect(self._ctrl.stop)
        self._btn_reset.clicked.connect(self._reset)
        self._sliders.value_changed.connect(self._on_slider)
        self._setup_env()

    def _setup_env(self):
        v = self._sliders.values()
        n, alpha, gamma, eps, speed = v[0], v[1]*0.05, v[2]*0.01, v[3]*0.01, v[4]
        self._ctrl.setup("gridworld", {"n": n},
                         "qlearning", {"alpha": alpha, "gamma": gamma, "epsilon": eps})
        self._ctrl.set_speed(speed)
        if self._ctrl.env: self._gw_view.load_env(self._ctrl.env)

    def _start(self): self._setup_env(); self._ctrl.start()
    def _reset(self):
        self._ctrl.reset(); self._curve.clear()
        self._status.set_text("Reset — press Train to start")

    def _on_agent(self, state, reward):
        self._gw_view.update_agent(state, reward)
        q = self._ctrl.get_q_table()
        if q is not None: self._gw_view.update_q(q)

    def _on_episode(self, ep, total_r):
        self._curve.add_point(total_r)
        self._status.set_text(f"Episode {ep}  |  Reward: {total_r:.2f}")

    def _on_slider(self, idx, val):
        if idx == 4: self._ctrl.set_speed(val)

    def set_phase(self, phase: float): self._gw_view.set_phase(phase)


class CartPoleTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctrl = TrainingController(self)
        self._ctrl.cartpole_state.connect(self._on_cp)
        self._ctrl.episode_done.connect(self._on_episode)

        lay = QHBoxLayout(self); lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(12)
        left = QVBoxLayout()
        self._cp_view = CartPoleView(); left.addWidget(self._cp_view)
        self._status = StatusBar(); left.addWidget(self._status)
        lay.addLayout(left, 3)

        right = QVBoxLayout(); right.setSpacing(10)
        self._sliders = SliderGroup([
            ("Alpha α",   1,  20,  2),
            ("Gamma γ",  50,  99, 95),
            ("Epsilon ε", 5,  50, 20),
            ("Speed",     1,  50, 10),
        ])
        panel = GlassPanel(); pl = panel.layout()
        pl.addWidget(QLabel("Hyperparameters")); pl.addWidget(self._sliders)
        right.addWidget(panel)

        self._curve = LearningCurve()
        cp = GlassPanel(); cl = cp.layout()
        cl.addWidget(QLabel("Episode Rewards")); cl.addWidget(self._curve)
        right.addWidget(cp)

        btn_lay = QHBoxLayout()
        self._btn_start = QPushButton("▶  Train")
        self._btn_stop  = QPushButton("■  Stop")
        self._btn_reset = QPushButton("↺  Reset")
        for btn in [self._btn_start, self._btn_stop, self._btn_reset]:
            btn.setFixedHeight(36); btn_lay.addWidget(btn)
        right.addLayout(btn_lay); right.addStretch()
        lay.addLayout(right, 2)

        self._btn_start.clicked.connect(self._start)
        self._btn_stop.clicked.connect(self._ctrl.stop)
        self._btn_reset.clicked.connect(self._reset)
        self._setup_env()

    def _setup_env(self):
        v = self._sliders.values()
        alpha, gamma, eps, speed = v[0]*0.05, v[1]*0.01, v[2]*0.01, v[3]
        self._ctrl.setup("cartpole", {},
                         "qlearning", {"alpha": alpha, "gamma": gamma, "epsilon": eps})
        self._ctrl.set_speed(speed)

    def _start(self): self._setup_env(); self._ctrl.start()
    def _reset(self): self._ctrl.reset(); self._curve.clear()

    def _on_cp(self, x, xd, th, thd, done):
        self._cp_view.update_state(x, xd, th, thd, done)

    def _on_episode(self, ep, total_r):
        self._curve.add_point(total_r)
        self._status.set_text(f"Episode {ep}  |  Reward: {total_r:.2f}")


class MazeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctrl = TrainingController(self)
        self._ctrl.maze_moved.connect(self._on_maze)
        self._ctrl.episode_done.connect(self._on_episode)

        lay = QHBoxLayout(self); lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(12)
        left = QVBoxLayout()
        self._maze_view = MazeView(); left.addWidget(self._maze_view)
        self._status = StatusBar(); left.addWidget(self._status)
        lay.addLayout(left, 3)

        right = QVBoxLayout(); right.setSpacing(10)
        self._sliders = SliderGroup([
            ("Rows",      3, 12,  5),
            ("Cols",      3, 12,  5),
            ("Alpha α",   1, 20,  2),
            ("Gamma γ",  50, 99, 95),
            ("Speed",     1, 50, 10),
        ])
        panel = GlassPanel(); pl = panel.layout()
        pl.addWidget(QLabel("Hyperparameters")); pl.addWidget(self._sliders)
        right.addWidget(panel)

        self._curve = LearningCurve()
        cp = GlassPanel(); cl = cp.layout()
        cl.addWidget(QLabel("Episode Rewards")); cl.addWidget(self._curve)
        right.addWidget(cp)

        btn_lay = QHBoxLayout()
        self._btn_start = QPushButton("▶  Train")
        self._btn_stop  = QPushButton("■  Stop")
        self._btn_reset = QPushButton("↺  Reset")
        for btn in [self._btn_start, self._btn_stop, self._btn_reset]:
            btn.setFixedHeight(36); btn_lay.addWidget(btn)
        right.addLayout(btn_lay); right.addStretch()
        lay.addLayout(right, 2)

        self._btn_start.clicked.connect(self._start)
        self._btn_stop.clicked.connect(self._ctrl.stop)
        self._btn_reset.clicked.connect(self._reset)
        self._setup_env()

    def _setup_env(self):
        v = self._sliders.values()
        rows, cols, alpha, gamma, speed = v[0], v[1], v[2]*0.05, v[3]*0.01, v[4]
        self._ctrl.setup("maze", {"rows": rows, "cols": cols},
                         "qlearning", {"alpha": alpha, "gamma": gamma, "epsilon": 0.2})
        self._ctrl.set_speed(speed)
        if self._ctrl.env: self._maze_view.load_maze(self._ctrl.env)

    def _start(self): self._setup_env(); self._ctrl.start()
    def _reset(self): self._ctrl.reset(); self._curve.clear()

    def _on_maze(self, pos, reward): self._maze_view.update_agent(pos)

    def _on_episode(self, ep, total_r):
        self._curve.add_point(total_r)
        self._status.set_text(f"Episode {ep}  |  Reward: {total_r:.2f}")


class PlaygroundView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabBar::tab{color:rgba(255,255,255,0.7);padding:8px 20px;"
            "background:transparent;border:none;}"
            "QTabBar::tab:selected{color:#00D4FF;"
            "border-bottom:2px solid #00D4FF;}"
        )
        self._gw_tab   = GridWorldTab()
        self._cp_tab   = CartPoleTab()
        self._maze_tab = MazeTab()
        self._tabs.addTab(self._gw_tab,   "GridWorld")
        self._tabs.addTab(self._cp_tab,   "CartPole")
        self._tabs.addTab(self._maze_tab, "Maze")
        lay.addWidget(self._tabs)

    def set_phase(self, phase: float): self._gw_tab.set_phase(phase)
