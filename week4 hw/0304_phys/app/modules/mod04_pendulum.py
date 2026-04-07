"""
MOD-04 — Pendulum Period Prediction
(L, θ₀) → T neural network + QPainter animation + phase space.
FR-MOD04-01..36  |  SRS-PHYSAI-004
"""
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                                QPushButton, QLabel, QComboBox, QSlider, QTabWidget)
from PySide6.QtGui import QKeySequence, QShortcut, QFont

from app.ml.data_generators import DataGenerators
from app.ml.models import ModelFactory
from app.ml.training_worker import TrainingConfig
from app.physics.pendulum import PendulumPhysics
from app.widgets.matplotlib_widget import MatplotlibWidget
from app.widgets.param_group import ParamGroup
from app.widgets.pendulum_canvas import PendulumCanvas, PendulumAnimationController
from .base_module import BaseModule, ModuleState


class PendulumModule(BaseModule):
    MODULE_ID   = "MOD-04"
    MODULE_NAME = "Pendulum Period Prediction"
    MODULE_DESC = "NN learns (L, θ₀) → T. Compares small-angle approx vs exact vs NN."

    def _setup_param_panel(self) -> QWidget:
        panel = QWidget(); lay = QVBoxLayout(panel)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)

        self._pg_pend = ParamGroup("Pendulum")
        self._L_sl  = self._pg_pend.add_slider('L', label="Length L",
            min_val=0.5, max_val=3.0, default=1.0, step=0.1, unit="m", decimals=1)
        self._th_sl = self._pg_pend.add_slider('theta0', label="Init Angle θ₀",
            min_val=5, max_val=80, default=30, step=1, unit="°", decimals=0)
        lay.addWidget(self._pg_pend)

        self._pg_train = ParamGroup("Training")
        self._n_sl  = self._pg_train.add_slider('n', label="Samples",
            min_val=200, max_val=5000, default=2000, step=100, decimals=0)
        self._ep_sl = self._pg_train.add_slider('epochs', label="Epochs",
            min_val=20, max_val=500, default=100, step=10, decimals=0)
        lay.addWidget(self._pg_train)

        self._L_sl.value_changed.connect(self._on_phys_param_changed)
        self._th_sl.value_changed.connect(self._on_phys_param_changed)
        self._pg_train.any_value_changed.connect(lambda *_: self._mark_dirty())

        # Period panel (FR-MOD04-05..10)
        self._period_group = ParamGroup("Period Comparison")
        self._T_small_lbl = self._period_group.add_label('T_small', "T_small: — s")
        self._T_exact_lbl = self._period_group.add_label('T_exact', "T_exact: — s")
        self._T_pred_lbl  = self._period_group.add_label('T_pred',  "T_pred:  — s")
        self._err_small_lbl = self._period_group.add_label('err_small', "")
        lay.addWidget(self._period_group)

        self._update_period_panel()
        return panel

    def _setup_extra_controls(self, lay):
        """FR-MOD04-26..29: Play/Pause + speed + isochronism toggle."""
        anim_row = QHBoxLayout()
        self._play_btn  = QPushButton("▶ Animate")
        self._pause_btn = QPushButton("⏸ Pause")
        self._play_btn.clicked.connect(self._on_play)
        self._pause_btn.clicked.connect(self._on_pause)
        anim_row.addWidget(self._play_btn); anim_row.addWidget(self._pause_btn)
        lay.addLayout(anim_row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed:"))
        self._speed_cb = QComboBox()
        self._speed_cb.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self._speed_cb.setCurrentIndex(2)
        self._speed_cb.currentIndexChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self._speed_cb)
        lay.addLayout(speed_row)

        # FR-MOD04-30..33: Isochronism demo
        self._iso_btn = QPushButton("Isochronism Demo")
        self._iso_btn.setCheckable(True)
        self._iso_btn.toggled.connect(self._on_iso_toggled)
        lay.addWidget(self._iso_btn)
        self._iso_lbl = QLabel("")
        self._iso_lbl.setWordWrap(True)
        self._iso_lbl.setStyleSheet("font-size:9pt;color:#757575;")
        lay.addWidget(self._iso_lbl)

    def _setup_plot_area(self) -> QWidget:
        # Right side: QTabWidget with Animation tab + Analysis tab
        self._plot_tabs = QTabWidget()

        # Tab 0: Animation view
        anim_widget = QWidget()
        anim_lay    = QVBoxLayout(anim_widget); anim_lay.setContentsMargins(0,0,0,0)
        h_split = QSplitter(Qt.Orientation.Horizontal)

        self.pendulum_canvas = PendulumCanvas()
        h_split.addWidget(self.pendulum_canvas)

        v_split = QSplitter(Qt.Orientation.Vertical)
        self.mpl_theta = MatplotlibWidget(figsize=(8, 4))
        self.mpl_phase = MatplotlibWidget(figsize=(8, 4))
        v_split.addWidget(self.mpl_theta); v_split.addWidget(self.mpl_phase)
        h_split.addWidget(v_split)
        h_split.setSizes([400, 600])
        anim_lay.addWidget(h_split)
        self._plot_tabs.addTab(anim_widget, "🎬  Animation")

        # Tab 1: Analysis (FR-MOD04-34..36)
        self.mpl_analysis = MatplotlibWidget(figsize=(12, 5))
        self._plot_tabs.addTab(self.mpl_analysis, "📊  Analysis")

        self._anim_ctrl = PendulumAnimationController(self.pendulum_canvas)

        # Space shortcut (FR-MOD04-26)
        self._space_sc = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self._space_sc.activated.connect(self._anim_ctrl.toggle)

        self._init_empty_plots()
        return self._plot_tabs

    def _init_empty_plots(self):
        ax = self.mpl_theta.fresh_axes()
        ax.set_title('θ(t) — run to simulate', fontweight='bold')
        ax.set_xlabel('t (s)'); ax.set_ylabel('θ (°)'); ax.grid(True, alpha=0.3)
        self.mpl_theta.draw()
        ax = self.mpl_phase.fresh_axes()
        ax.set_title('Phase Space', fontweight='bold')
        ax.set_xlabel('θ (°)'); ax.set_ylabel('ω (°/s)'); ax.grid(True, alpha=0.3)
        self.mpl_phase.draw()

    # ── BaseModule interface ──────────────────────────────────────────────────
    def _build_model(self): return ModelFactory.pendulum_period()

    def _generate_data(self):
        X, y = DataGenerators.pendulum(int(self._n_sl.value))
        return X, y

    def _get_training_config(self) -> TrainingConfig:
        return TrainingConfig(epochs=int(self._ep_sl.value),
                              validation_split=0.2, log_interval=10,
                              use_reduce_lr=True, reduce_lr_patience=20, reduce_lr_factor=0.5)

    def _on_training_finished_impl(self, model, history):
        self._update_period_panel()
        self._run_simulation_and_animate()

    def _on_phys_param_changed(self, *_):
        """FR-MOD04-08,14: update period panel and re-simulate immediately."""
        self._update_period_panel()
        if self._state == ModuleState.TRAINED:
            self._run_simulation_and_animate()
        # FR-MOD04-33: disable iso if theta changed
        if self._iso_btn.isChecked():
            self._iso_btn.setChecked(False)

    # ── Period panel ──────────────────────────────────────────────────────────
    def _update_period_panel(self):
        """FR-MOD04-05..09: update T_small, T_exact, T_pred, error highlights."""
        L     = self._L_sl.value
        th    = self._th_sl.value
        phys  = PendulumPhysics(L)
        T_sm  = phys.small_angle_period()
        T_ex  = PendulumPhysics.true_period(L, th)
        err_sm = abs(T_sm - T_ex) / T_ex * 100

        self._T_small_lbl.setText(f"T_small: {T_sm:.3f} s")
        self._T_exact_lbl.setText(f"T_exact: {T_ex:.3f} s")

        # FR-MOD04-09: amber if θ₀ > 30°
        if th > 30:
            self._err_small_lbl.setText(f"⚠ Small-angle error: {err_sm:.1f}%")
            self._err_small_lbl.setStyleSheet("color:#FB8C00;font-size:9pt;font-weight:bold;")
        else:
            self._err_small_lbl.setText(f"Small-angle error: {err_sm:.2f}%")
            self._err_small_lbl.setStyleSheet("color:#757575;font-size:9pt;")

        if self._model is not None:
            T_pred = float(self._model.predict([[L, th]], verbose=0)[0, 0])
            err_pr = abs(T_pred - T_ex) / T_ex * 100
            self._T_pred_lbl.setText(f"T_pred:  {T_pred:.3f} s  (err: {err_pr:.2f}%)")
        else:
            self._T_pred_lbl.setText("T_pred:  — s  (train first)")

    # ── Simulation + animation ────────────────────────────────────────────────
    def _run_simulation_and_animate(self):
        """FR-MOD04-11..14, FR-MOD04-28: RK4 simulate + reload animation."""
        L   = self._L_sl.value
        th  = self._th_sl.value
        T_ex = PendulumPhysics.true_period(L, th)
        phys = PendulumPhysics(L)
        traj = phys.simulate(th, n_periods=3, dt=0.01)
        # t array
        dt   = 0.01
        t_arr = np.arange(len(traj)) * dt
        theta_deg = np.degrees(traj[:, 0])
        omega_deg = np.degrees(traj[:, 1])

        # θ(t) plot — FR-MOD04-15..18
        ax = self.mpl_theta.fresh_axes()
        T_pred = float(self._model.predict([[L, th]], verbose=0)[0, 0]) if self._model else 0
        ax.plot(t_arr, theta_deg, 'b-', lw=2.0)
        ax.axhline(0, color='k', ls='--', lw=1, alpha=0.5)
        for k in range(1, 4):
            ax.axvline(k * T_ex, color='r', ls='--', lw=1.5, alpha=0.7,
                       label='Period' if k == 1 else None)
        ax.set_xlabel('t (s)'); ax.set_ylabel('θ (°)')
        ax.set_title(f'θ₀={th:.0f}°  T_exact={T_ex:.3f}s  T_pred={T_pred:.3f}s',
                     fontweight='bold', fontsize=9)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        self.mpl_theta.draw()

        # Phase space — FR-MOD04-19..21
        ax2 = self.mpl_phase.fresh_axes()
        ax2.plot(theta_deg, omega_deg, 'g-', lw=1.5, alpha=0.7)
        ax2.scatter([theta_deg[0]], [omega_deg[0]], c='r', s=80, zorder=5, label='Start')
        ax2.set_xlabel('θ (°)'); ax2.set_ylabel('ω (°/s)')
        ax2.set_title('Phase Space (θ vs ω)', fontweight='bold')
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
        self.mpl_phase.draw()

        # Load animation — FR-MOD04-28: auto-start
        traj2 = None
        if self._iso_btn.isChecked():
            phys2 = PendulumPhysics(L)
            traj2 = phys2.simulate(5.0, n_periods=3, dt=0.01)  # FR-MOD04-30: θ₀=5°
            self._update_iso_label(L, th)
        self._anim_ctrl.load(traj, L, traj2=traj2)
        self._anim_ctrl.play()

    def _update_iso_label(self, L: float, th: float):
        T1 = PendulumPhysics.true_period(L, th)
        T2 = PendulumPhysics.true_period(L, 5.0)
        dT = abs(T1 - T2)
        pct = dT / T1 * 100
        self._iso_lbl.setText(f"ΔT = {dT:.4f} s ({pct:.2f}% difference)")

    # ── Analysis tab ──────────────────────────────────────────────────────────
    def _render_analysis(self):
        """FR-MOD04-34..36: MAPE vs θ₀ + True vs Pred scatter."""
        if self._model is None: return
        axes = self.mpl_analysis.fresh_axes(1, 2)
        ax_mape, ax_scatter = axes

        # MAPE vs θ₀ for L=1.0m
        L_fixed = 1.0
        thetas  = np.linspace(5, 80, 50)
        T_true  = np.array([PendulumPhysics.true_period(L_fixed, t) for t in thetas])
        X_grid  = np.column_stack([np.full(50, L_fixed), thetas])
        T_pred  = self._model.predict(X_grid, verbose=0).flatten()
        mape    = np.abs(T_pred - T_true) / T_true * 100

        ax_mape.plot(thetas, mape, 'b-', lw=2)
        ax_mape.axhline(1.0, color='r', ls='--', lw=1.5, label='1% target')
        ax_mape.set_xlabel('θ₀ (°)'); ax_mape.set_ylabel('MAPE (%)')
        ax_mape.set_title('MAPE vs θ₀  (L=1.0m)', fontweight='bold')
        ax_mape.legend(fontsize=9); ax_mape.grid(True, alpha=0.3)

        # True vs Pred scatter
        rng = np.random.default_rng(seed=1)
        L_t = rng.uniform(0.5, 3.0, 200); th_t = rng.uniform(5, 80, 200)
        T_true2 = np.array([PendulumPhysics.true_period(l, t) for l, t in zip(L_t, th_t)])
        T_pred2 = self._model.predict(np.column_stack([L_t, th_t]), verbose=0).flatten()
        ax_scatter.scatter(T_true2, T_pred2, alpha=0.5, s=15, c='C0')
        mn, mx = T_true2.min(), T_true2.max()
        ax_scatter.plot([mn, mx], [mn, mx], 'k--', lw=1.5, label='y=x')
        ax_scatter.set_xlabel('T_exact (s)'); ax_scatter.set_ylabel('T_pred (s)')
        ax_scatter.set_title('True vs Predicted Period', fontweight='bold')
        ax_scatter.legend(fontsize=9); ax_scatter.grid(True, alpha=0.3)
        self.mpl_analysis.draw()

    # ── Animation controls ────────────────────────────────────────────────────
    def _on_play(self): self._anim_ctrl.play()
    def _on_pause(self): self._anim_ctrl.pause()

    def _on_speed_changed(self, idx):
        speeds = [0.25, 0.5, 1.0, 2.0, 4.0]
        self._anim_ctrl.set_speed(speeds[idx])

    def _on_iso_toggled(self, checked):
        if checked and self._state == ModuleState.TRAINED:
            self._run_simulation_and_animate()
        elif not checked:
            self._iso_lbl.setText("")
            self._anim_ctrl.load(self._anim_ctrl._traj, self._L_sl.value)

    def _on_reset_impl(self):
        self._anim_ctrl.pause()
        self._iso_btn.setChecked(False)
        self._iso_lbl.setText("")
        self._init_empty_plots()
        self._update_period_panel()

    def _on_training_finished(self, model, history):
        super()._on_training_finished(model, history)
        self._render_analysis()

    def get_param_values(self):
        return {**self._pg_pend.values(), **self._pg_train.values()}

    def get_metrics(self):
        if self._model is None: return {}
        L = self._L_sl.value; th = self._th_sl.value
        T_ex   = PendulumPhysics.true_period(L, th)
        T_pred = float(self._model.predict([[L, th]], verbose=0)[0, 0])
        return {'T_exact': T_ex, 'T_pred': T_pred,
                'err_pct': abs(T_pred - T_ex) / T_ex * 100}

    def cleanup(self):
        self._anim_ctrl.pause()
        super().cleanup()
