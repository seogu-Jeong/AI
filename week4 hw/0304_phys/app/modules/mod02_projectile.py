"""
MOD-02 — Projectile Motion Regression
(v₀, θ, t) → (x, y) neural network regressor.
FR-MOD02-01..17  |  SRS-PHYSAI-003 §4.1–4.4
"""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel)

from app.ml.data_generators import DataGenerators
from app.ml.models import ModelFactory
from app.ml.training_worker import TrainingConfig
from app.widgets.matplotlib_widget import MatplotlibWidget
from app.widgets.param_group import ParamGroup
from .base_module import BaseModule

# tab10 palette for multi-condition overlay
TAB10 = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
G = 9.81


class ProjectileMotionModule(BaseModule):
    MODULE_ID   = "MOD-02"
    MODULE_NAME = "Projectile Motion Regression"
    MODULE_DESC = "NN learns (v₀, θ, t) → (x, y) under vacuum. Predicts full trajectory."

    def _setup_param_panel(self) -> QWidget:
        panel = QWidget(); lay = QVBoxLayout(panel)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)

        self._pg_launch = ParamGroup("Launch Parameters")
        self._v0_sl = self._pg_launch.add_slider('v0', label="Init Velocity v₀",
            min_val=10, max_val=50, default=30, step=1, unit="m/s", decimals=0)
        self._th_sl = self._pg_launch.add_slider('theta', label="Launch Angle θ",
            min_val=20, max_val=70, default=45, step=1, unit="°", decimals=0)
        lay.addWidget(self._pg_launch)

        self._pg_train = ParamGroup("Training Config")
        self._n_sl   = self._pg_train.add_slider('n', label="Train Samples",
            min_val=200, max_val=5000, default=2000, step=100, decimals=0)
        self._sig_sl = self._pg_train.add_slider('noise', label="Noise σ",
            min_val=0.0, max_val=2.0, default=0.5, step=0.1, decimals=1, unit="m")
        self._ep_sl  = self._pg_train.add_slider('epochs', label="Epochs",
            min_val=20, max_val=500, default=100, step=10, decimals=0)
        lay.addWidget(self._pg_train)

        self._v0_sl.value_changed.connect(lambda _: self._on_slider_changed())
        self._th_sl.value_changed.connect(lambda _: self._on_slider_changed())
        self._pg_train.any_value_changed.connect(lambda *_: self._mark_dirty())
        return panel

    def _setup_extra_controls(self, lay):
        """FR-MOD02-14..17: multi-condition overlay buttons."""
        self._conditions = []
        btn_row = QHBoxLayout()
        self._add_cond_btn  = QPushButton("☆ Add Condition")
        self._clr_cond_btn  = QPushButton("Clear Conditions")
        self._add_cond_btn.clicked.connect(self._add_condition)
        self._clr_cond_btn.clicked.connect(self._clear_conditions)
        btn_row.addWidget(self._add_cond_btn); btn_row.addWidget(self._clr_cond_btn)
        lay.addLayout(btn_row)
        self._cond_lbl = QLabel("Active conditions: 0/5")
        self._cond_lbl.setStyleSheet("font-size:9pt;color:#757575;")
        lay.addWidget(self._cond_lbl)

    def _setup_plot_area(self) -> QWidget:
        self.mpl = MatplotlibWidget(figsize=(14, 9))
        self._setup_plots()
        return self.mpl

    def _setup_plots(self):
        """GridSpec 2×2, height_ratios=[2,1]. Pre-create artists."""
        gs = self.mpl.fresh_gridspec(2, 2, height_ratios=[2,1], hspace=0.35)
        fig = self.mpl.figure
        self._ax_traj = fig.add_subplot(gs[0, :])
        self._ax_xt   = fig.add_subplot(gs[1, 0])
        self._ax_yt   = fig.add_subplot(gs[1, 1])

        # Trajectory (FR-MOD02-11: xlim≥0, ylim≥0)
        self._ax_traj.set_xlabel('x (m)'); self._ax_traj.set_ylabel('y (m)')
        self._ax_traj.set_title('Trajectory Comparison', fontweight='bold')
        self._ax_traj.set_xlim(left=0); self._ax_traj.set_ylim(bottom=0)
        self._ax_traj.grid(True, alpha=0.3)
        self._line_traj_true, = self._ax_traj.plot([], [], 'b-',  lw=2.5, label='True')
        self._line_traj_pred, = self._ax_traj.plot([], [], 'r--', lw=2.0, label='NN Pred')
        self._ax_traj.legend(fontsize=9)
        self._cond_lines = []   # overlay lines

        # x(t)
        self._ax_xt.set_xlabel('t (s)'); self._ax_xt.set_ylabel('x (m)')
        self._ax_xt.set_title('x(t)', fontweight='bold'); self._ax_xt.grid(True, alpha=0.3)
        self._line_xt_true, = self._ax_xt.plot([], [], 'b-',  lw=2.0, label='True')
        self._line_xt_pred, = self._ax_xt.plot([], [], 'r--', lw=2.0, label='Pred')
        self._ax_xt.legend(fontsize=9)

        # y(t)
        self._ax_yt.set_xlabel('t (s)'); self._ax_yt.set_ylabel('y (m)')
        self._ax_yt.set_title('y(t)', fontweight='bold'); self._ax_yt.grid(True, alpha=0.3)
        self._line_yt_true, = self._ax_yt.plot([], [], 'b-',  lw=2.0, label='True')
        self._line_yt_pred, = self._ax_yt.plot([], [], 'r--', lw=2.0, label='Pred')
        self._ax_yt.legend(fontsize=9)
        self.mpl.draw()

    # ── BaseModule interface ──────────────────────────────────────────────────
    def _build_model(self): return ModelFactory.projectile_regression()

    def _generate_data(self):
        n    = int(self._n_sl.value)
        sig  = self._sig_sl.value
        X, Y = DataGenerators.projectile(n, noise_m=sig)
        self._X_train = X; self._Y_train = Y
        return X, Y

    def _get_training_config(self) -> TrainingConfig:
        return TrainingConfig(epochs=int(self._ep_sl.value),
                              validation_split=0.2, log_interval=10)

    def _on_progress(self, epoch, loss, val_loss):
        self._progress.update(epoch, self._get_training_config().epochs, loss, val_loss)

    def _on_training_finished_impl(self, model, history):
        self._update_trajectory(self._v0_sl.value, self._th_sl.value)

    def _on_slider_changed(self):
        """FR-MOD02-10: re-predict without retrain when TRAINED."""
        if self._state.name in ('TRAINED', 'DIRTY') and self._model is not None:
            self._update_trajectory(self._v0_sl.value, self._th_sl.value)

    def _update_trajectory(self, v0: float, theta: float):
        """FR-MOD02-09,10,11,12: 50-point prediction + annotate."""
        theta_rad = np.deg2rad(theta)
        t_flight  = 2 * v0 * np.sin(theta_rad) / G
        t_pts     = np.linspace(0, t_flight, 50)

        # True (analytical)
        x_true = v0 * np.cos(theta_rad) * t_pts
        y_true = v0 * np.sin(theta_rad) * t_pts - 0.5 * G * t_pts**2

        # NN prediction
        X_in  = np.column_stack([np.full(50, v0), np.full(50, theta), t_pts])
        pred  = self._model.predict(X_in, verbose=0)
        x_pred, y_pred = pred[:, 0], pred[:, 1]

        for line, xd, yd in [
            (self._line_traj_true, x_true, y_true),
            (self._line_traj_pred, x_pred, y_pred),
            (self._line_xt_true,   t_pts,  x_true),
            (self._line_xt_pred,   t_pts,  x_pred),
            (self._line_yt_true,   t_pts,  y_true),
            (self._line_yt_pred,   t_pts,  y_pred),
        ]:
            line.set_data(xd, yd)

        # FR-MOD02-12: annotate max height and max range
        for ax in (self._ax_traj, self._ax_xt, self._ax_yt):
            ax.relim(); ax.autoscale_view()
        for ann in getattr(self, '_annotations', []): ann.remove()
        self._annotations = []
        h_true = float(np.max(y_true)); r_true = float(x_true[-1])
        h_pred = float(np.max(y_pred)); r_pred = float(x_pred[np.argmax(y_pred)])
        self._annotations.append(self._ax_traj.annotate(
            f'H_true={h_true:.1f}m', xy=(x_true[np.argmax(y_true)], h_true),
            fontsize=8, color='blue', ha='center',
            arrowprops=dict(arrowstyle='->', color='blue'), xytext=(0, 15), textcoords='offset points'))
        self._annotations.append(self._ax_traj.annotate(
            f'R_true={r_true:.1f}m', xy=(r_true, 0),
            fontsize=8, color='blue', xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='blue')))

        title = (f"v₀={v0:.0f} m/s  θ={theta:.0f}°\n"
                 f"True: H={h_true:.1f}m R={r_true:.1f}m  |  Pred: H={h_pred:.1f}m R={r_pred:.1f}m")
        self._ax_traj.set_title(title, fontweight='bold', fontsize=9)
        self._ax_traj.set_xlim(left=0); self._ax_traj.set_ylim(bottom=0)
        self.mpl.draw_idle()

    # ── Multi-condition overlay ───────────────────────────────────────────────
    def _add_condition(self):
        """FR-MOD02-14..16: add current (v₀,θ) as overlay condition."""
        if len(self._conditions) >= 5 or self._model is None: return
        v0 = self._v0_sl.value; theta = self._th_sl.value
        n  = len(self._conditions)
        color = TAB10[n % len(TAB10)]
        theta_rad = np.deg2rad(theta)
        t_flight  = 2 * v0 * np.sin(theta_rad) / G
        t_pts     = np.linspace(0, t_flight, 50)
        x_true    = v0 * np.cos(theta_rad) * t_pts
        y_true    = v0 * np.sin(theta_rad) * t_pts - 0.5 * G * t_pts**2
        # FR-MOD02-15: legend label
        lbl = f"Cond {n+1}: v₀={v0:.0f}m/s, θ={theta:.0f}°"
        line, = self._ax_traj.plot(x_true, y_true, '-', color=color, lw=1.5,
                                   alpha=0.7, label=lbl)
        self._cond_lines.append(line)
        self._conditions.append((v0, theta))
        self._ax_traj.legend(fontsize=8)
        self._cond_lbl.setText(f"Active conditions: {len(self._conditions)}/5")
        # FR-MOD02-16: disable at limit
        self._add_cond_btn.setEnabled(len(self._conditions) < 5)
        self.mpl.draw_idle()

    def _clear_conditions(self):
        """FR-MOD02-17: remove all overlay conditions."""
        for line in self._cond_lines: line.remove()
        self._cond_lines.clear(); self._conditions.clear()
        self._cond_lbl.setText("Active conditions: 0/5")
        self._add_cond_btn.setEnabled(True)
        self._ax_traj.legend(fontsize=9)
        self.mpl.draw_idle()

    def _on_reset_impl(self):
        self._conditions.clear()
        for line in self._cond_lines: line.remove()
        self._cond_lines.clear()
        self._cond_lbl.setText("Active conditions: 0/5")
        self._add_cond_btn.setEnabled(True)
        self._setup_plots()

    def get_param_values(self):
        return {**self._pg_launch.values(), **self._pg_train.values()}
