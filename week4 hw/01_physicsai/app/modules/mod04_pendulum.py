"""
MOD-04 — Pendulum Simulation
QPainter animation + NN period prediction + θ(t) + phase space.
"""
import math
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QComboBox,
)
from PySide6.QtGui import QFont

from app.physics.pendulum import PendulumPhysics
from app.ml.data_generators import DataGenerators
from app.ml.models import ModelFactory
from app.ml.training_worker import TrainingConfig
from app.widgets.matplotlib_widget import MatplotlibWidget
from app.widgets.param_group import ParamGroup
from app.widgets.pendulum_canvas import PendulumCanvas
from .base_module import BaseModule


class PendulumModule(BaseModule):
    MODULE_ID   = "MOD-04"
    MODULE_NAME = "Pendulum Simulation"
    MODULE_DESC = "RK4 integration + NN period prediction + live QPainter animation"

    # ── parameter panel ──────────────────────────────────────────────────────
    def _setup_param_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._pg_phys = ParamGroup("Physical Parameters")
        self._L_sl     = self._pg_phys.add_slider('L', label="Length L", min_val=0.3, max_val=3.0,
            default=1.0, step=0.1, unit="m", decimals=1,
            tooltip="Pendulum arm length. T_small = 2π√(L/g)")
        self._th_sl    = self._pg_phys.add_slider('theta0', label="Initial Angle θ₀",
            min_val=5, max_val=80, default=30, step=5, unit="°", decimals=0,
            tooltip="Initial displacement. > 30° makes small-angle approx. break down")
        layout.addWidget(self._pg_phys)

        self._pg_tr = ParamGroup("Training")
        self._ep_sl = self._pg_tr.add_slider('epochs', label="Epochs",
            min_val=50, max_val=500, default=150, step=50, decimals=0)
        self._n_sl  = self._pg_tr.add_slider('n_samples', label="Samples",
            min_val=500, max_val=5000, default=2000, step=500, decimals=0)
        layout.addWidget(self._pg_tr)

        # Period summary panel
        self._period_box = ParamGroup("Period Analysis")
        self._T_small_lbl = QLabel("T_small  = —")
        self._T_exact_lbl = QLabel("T_exact  = —")
        self._T_pred_lbl  = QLabel("T_pred   = — (not trained)")
        self._T_err_lbl   = QLabel("")
        bold = QFont(); bold.setFamily('Courier New'); bold.setPointSize(10)
        for lbl in (self._T_small_lbl, self._T_exact_lbl, self._T_pred_lbl):
            lbl.setFont(bold)
            self._period_box.layout().addWidget(lbl)
        self._period_box.layout().addWidget(self._T_err_lbl)
        layout.addWidget(self._period_box)

        # Animation controls
        self._pg_anim = ParamGroup("Animation")
        self._anim_speed_cb = self._pg_anim.add_combo(
            'speed', "Speed:", ['0.25×', '0.5×', '1×', '2×', '4×'], default_idx=2,
            tooltip="Animation playback speed multiplier")
        layout.addWidget(self._pg_anim)

        # Play / Pause button
        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle_animation)
        layout.addWidget(self._play_btn)

        # Update period display live
        self._pg_phys.any_value_changed.connect(self._update_period_display)
        self._pg_phys.any_value_changed.connect(lambda *_: self._mark_dirty())
        self._pg_tr.any_value_changed.connect(lambda *_: self._mark_dirty())
        self._anim_speed_cb.currentTextChanged.connect(self._on_speed_changed)

        self._update_period_display()
        return panel

    # ── plot area ────────────────────────────────────────────────────────────
    def _setup_plot_area(self) -> QWidget:
        # H-splitter: pendulum canvas (left) | plots (right)
        h_split = QSplitter(Qt.Orientation.Horizontal)

        self.pend_canvas = PendulumCanvas()
        h_split.addWidget(self.pend_canvas)

        v_split = QSplitter(Qt.Orientation.Vertical)
        self.mpl_theta = MatplotlibWidget(figsize=(8, 3))
        self.mpl_phase = MatplotlibWidget(figsize=(8, 3))
        v_split.addWidget(self.mpl_theta)
        v_split.addWidget(self.mpl_phase)
        h_split.addWidget(v_split)
        h_split.setSizes([380, 620])

        self._ax_theta = self.mpl_theta.fresh_axes()
        self._ax_theta.set_title('Angular Displacement θ(t)', fontweight='bold')
        self._ax_theta.set_xlabel('Time (s)'); self._ax_theta.set_ylabel('θ (°)')
        self._ax_theta.grid(True, alpha=0.25)
        self.mpl_theta.draw()

        self._ax_phase = self.mpl_phase.fresh_axes()
        self._ax_phase.set_title('Phase Space  θ vs ω', fontweight='bold')
        self._ax_phase.set_xlabel('θ (°)'); self._ax_phase.set_ylabel('ω (°/s)')
        self._ax_phase.grid(True, alpha=0.25)
        self.mpl_phase.draw()

        # Animation timer
        self._anim_timer  = QTimer()
        self._anim_timer.timeout.connect(self._anim_frame)
        self._anim_traj   = None
        self._anim_idx    = 0
        self._anim_speed  = 1.0
        self._anim_dt     = 0.01

        return h_split

    # ── physics helpers ──────────────────────────────────────────────────────
    def _update_period_display(self, *_):
        L  = self._L_sl.value
        th = self._th_sl.value
        T_small = 2 * math.pi * math.sqrt(L / 9.81)
        T_exact = PendulumPhysics.true_period(L, th)
        err_pct = abs(T_small - T_exact) / T_exact * 100

        self._T_small_lbl.setText(f"T_small = {T_small:.4f} s")
        self._T_exact_lbl.setText(f"T_exact = {T_exact:.4f} s")

        if err_pct > 3.0:
            self._T_small_lbl.setStyleSheet("color: #FB8C00; font-weight: bold;")
            self._T_err_lbl.setText(f"⚠ Small-angle error: {err_pct:.1f}%")
            self._T_err_lbl.setStyleSheet("color: #FB8C00;")
        else:
            self._T_small_lbl.setStyleSheet("")
            self._T_err_lbl.setText(f"Small-angle error: {err_pct:.2f}%")
            self._T_err_lbl.setStyleSheet("color: #9E9E9E;")

    # ── animation ────────────────────────────────────────────────────────────
    def _toggle_animation(self):
        if self._anim_timer.isActive():
            self._anim_timer.stop()
            self._play_btn.setText("▶  Play")
        else:
            self._anim_timer.start(max(1, int(33 / self._anim_speed)))
            self._play_btn.setText("⏸  Pause")

    def _on_speed_changed(self, txt: str):
        mult = float(txt.replace('×', ''))
        self._anim_speed = mult
        if self._anim_timer.isActive():
            self._anim_timer.setInterval(max(1, int(33 / mult)))

    def _anim_frame(self):
        if self._anim_traj is None:
            return
        N = len(self._anim_traj)
        if self._anim_idx >= N:
            self._anim_idx = 0
        state = self._anim_traj[self._anim_idx]
        self.pend_canvas.set_state(float(state[0]), float(state[1]))
        self._anim_idx += 1

    # ── BaseModule interface ─────────────────────────────────────────────────
    def _build_model(self):
        return ModelFactory.pendulum_period()

    def _generate_data(self):
        n = int(self._n_sl.value)
        X, y = DataGenerators.pendulum(n_samples=n)
        return X, y

    def _get_training_config(self) -> TrainingConfig:
        return TrainingConfig(epochs=int(self._ep_sl.value))

    def _on_training_finished_impl(self, model, history):
        L  = self._L_sl.value
        th = self._th_sl.value
        X_in  = np.array([[L, th]])
        T_pred = float(model.predict(X_in, verbose=0)[0, 0])
        T_exact = PendulumPhysics.true_period(L, th)
        err_pct = abs(T_pred - T_exact) / T_exact * 100

        self._T_pred_lbl.setText(f"T_pred  = {T_pred:.4f} s  (err {err_pct:.2f}%)")
        if err_pct < 2.0:
            self._T_pred_lbl.setStyleSheet("color: #43A047; font-weight: bold;")
        else:
            self._T_pred_lbl.setStyleSheet("color: #E53935;")

        # Simulate trajectory for plots + animation
        physics = PendulumPhysics(L=L)
        traj    = physics.simulate(th, n_periods=4.0, dt=0.01)
        t_arr   = np.arange(len(traj)) * 0.01
        th_deg  = np.degrees(traj[:, 0])
        om_deg  = np.degrees(traj[:, 1])

        # θ(t) plot
        ax = self.mpl_theta.fresh_axes()
        ax.plot(t_arr, th_deg, 'C0-', lw=2.0)
        ax.axhline(0, color='grey', ls='--', lw=0.8, alpha=0.5)
        for k in range(1, 6):
            ax.axvline(k * T_exact, color='C3', ls='--', lw=1.2, alpha=0.65,
                       label='T_exact' if k == 1 else None)
            ax.axvline(k * T_pred,  color='C2', ls=':',  lw=1.2, alpha=0.65,
                       label='T_pred' if k == 1 else None)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('θ (°)')
        ax.set_title(f'θ₀={th}°  T_exact={T_exact:.3f}s  T_pred={T_pred:.3f}s', fontweight='bold', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        self.mpl_theta.draw()

        # Phase space
        ax2 = self.mpl_phase.fresh_axes()
        sc = ax2.scatter(th_deg, om_deg, c=t_arr, cmap='plasma', s=4, alpha=0.7)
        ax2.scatter([th_deg[0]], [om_deg[0]], c='C3', s=80, zorder=5, label='Start')
        ax2.set_xlabel('θ (°)'); ax2.set_ylabel('ω (°/s)')
        ax2.set_title('Phase Space', fontweight='bold')
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)
        self.mpl_phase.figure.colorbar(sc, ax=ax2, label='Time (s)')
        self.mpl_phase.draw()

        # Load animation
        self._anim_traj = traj
        self._anim_idx  = 0
        self.pend_canvas.set_pendulum_length(L)
        self.pend_canvas.set_energy_reference(
            physics.energy(traj[0]))
        self._play_btn.setEnabled(True)
        # Auto-play
        self._anim_timer.start(33)
        self._play_btn.setText("⏸  Pause")

    def _on_reset_impl(self):
        self._anim_timer.stop()
        self._anim_traj = None
        self._play_btn.setEnabled(False)
        self._play_btn.setText("▶  Play")
        self.pend_canvas.set_state(0.0)
        self._ax_theta = self.mpl_theta.fresh_axes()
        self._ax_theta.set_title('Angular Displacement θ(t)', fontweight='bold')
        self._ax_theta.set_xlabel('Time (s)'); self._ax_theta.set_ylabel('θ (°)')
        self._ax_theta.grid(True, alpha=0.25)
        self.mpl_theta.draw()
        self._ax_phase = self.mpl_phase.fresh_axes()
        self._ax_phase.set_title('Phase Space', fontweight='bold')
        self._ax_phase.grid(True, alpha=0.25)
        self.mpl_phase.draw()
        self._T_pred_lbl.setText("T_pred   = — (not trained)")
        self._T_pred_lbl.setStyleSheet("")

    def cleanup(self):
        self._anim_timer.stop()
        super().cleanup()

    def get_param_values(self):
        return {**self._pg_phys.values(), **self._pg_tr.values()}

    def get_metrics(self):
        return {}
