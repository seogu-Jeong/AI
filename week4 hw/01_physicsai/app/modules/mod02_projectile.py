"""
MOD-02 — Projectile Motion Regression
Neural network learns (v₀, θ, t) → (x, y) mapping.
"""
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout

from app.ml.data_generators import DataGenerators
from app.ml.models import ModelFactory
from app.ml.training_worker import TrainingConfig
from app.widgets.matplotlib_widget import MatplotlibWidget
from app.widgets.param_group import ParamGroup
from .base_module import BaseModule


class ProjectileRegressionModule(BaseModule):
    MODULE_ID   = "MOD-02"
    MODULE_NAME = "Projectile Motion"
    MODULE_DESC = "NN learns trajectory (v₀, θ, t) → (x, y) from RK4-generated data"

    def _setup_param_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._pg_sim = ParamGroup("Simulation Parameters")
        self._v0_sl  = self._pg_sim.add_slider('v0', label="Initial Speed", min_val=10, max_val=50,
            default=30, step=1, unit="m/s", decimals=0,
            tooltip="Initial projectile speed (10–50 m/s)")
        self._th_sl  = self._pg_sim.add_slider('theta', label="Launch Angle", min_val=20, max_val=70,
            default=45, step=1, unit="°", decimals=0,
            tooltip="Launch angle above horizontal (20–70°)")
        layout.addWidget(self._pg_sim)

        self._pg_tr = ParamGroup("Training")
        self._ep_sl = self._pg_tr.add_slider('epochs', label="Epochs", min_val=50, max_val=1000,
            default=300, step=50, unit="", decimals=0)
        self._n_sl  = self._pg_tr.add_slider('n_samples', label="Samples", min_val=500, max_val=5000,
            default=2000, step=500, unit="", decimals=0)
        layout.addWidget(self._pg_tr)

        for pg in (self._pg_sim, self._pg_tr):
            pg.any_value_changed.connect(lambda *_: self._mark_dirty())
        return panel

    def _setup_plot_area(self) -> QWidget:
        self.mpl = MatplotlibWidget(figsize=(14, 9))
        self._epochs_buf, self._losses_buf = [], []
        gs = self.mpl.fresh_gridspec(2, 2, height_ratios=[2, 1], hspace=0.42, wspace=0.35)
        fig = self.mpl.figure
        self._ax_traj = fig.add_subplot(gs[0, :])
        self._ax_xt   = fig.add_subplot(gs[1, 0])
        self._ax_yt   = fig.add_subplot(gs[1, 1])
        self._init_artists()
        return self.mpl

    def _init_artists(self):
        self._ax_traj.set_title('Trajectory', fontweight='bold')
        self._ax_traj.set_xlabel('x (m)'); self._ax_traj.set_ylabel('y (m)')
        self._ax_traj.grid(True, alpha=0.25)
        self._ax_xt.set_xlabel('t (s)'); self._ax_xt.set_ylabel('x (m)')
        self._ax_xt.grid(True, alpha=0.25)
        self._ax_yt.set_xlabel('t (s)'); self._ax_yt.set_ylabel('y (m)')
        self._ax_yt.grid(True, alpha=0.25)

        self._line_traj_t, = self._ax_traj.plot([], [], 'C0-',  lw=2.5, label='True (analytic)')
        self._line_traj_p, = self._ax_traj.plot([], [], 'C3--', lw=2.0, label='NN Prediction')
        self._ax_traj.legend(fontsize=9)

        self._loss_line,   = self._ax_xt.plot([], [], 'C2-', lw=1.8)
        self._line_xt_t,   = self._ax_xt.plot([], [], 'C0-', lw=2.0, label='True')
        self._line_xt_p,   = self._ax_xt.plot([], [], 'C3--', lw=2.0, label='Pred')
        self._ax_xt.legend(fontsize=9)

        self._line_yt_t,   = self._ax_yt.plot([], [], 'C0-', lw=2.0, label='True')
        self._line_yt_p,   = self._ax_yt.plot([], [], 'C3--', lw=2.0, label='Pred')
        self._ax_yt.legend(fontsize=9)

        self.mpl.draw()

    # ── BaseModule interface ─────────────────────────────────────────────────
    def _build_model(self):
        return ModelFactory.projectile_regression()

    def _generate_data(self):
        n = int(self._n_sl.value)
        X, y = DataGenerators.projectile(n_samples=n)
        return X, y

    def _get_training_config(self) -> TrainingConfig:
        return TrainingConfig(epochs=int(self._ep_sl.value), batch_size=64)

    def _on_progress(self, epoch: int, loss: float, val_loss: float):
        super()._on_progress(epoch, loss, val_loss)
        cfg = self._get_training_config()
        self._epochs_buf.append(epoch)
        self._losses_buf.append(loss)
        self._loss_line.set_data(self._epochs_buf, self._losses_buf)
        self._ax_xt.relim(); self._ax_xt.autoscale_view()
        self._ax_xt.set_title(f'x(t) — Loss {loss:.4f}', fontweight='bold', fontsize=9)
        self.mpl.draw_idle()

    def _on_training_finished_impl(self, model, history):
        v0    = self._v0_sl.value
        theta = self._th_sl.value
        self._update_trajectory(model, v0, theta, history)

    def _update_trajectory(self, model, v0, theta, history=None):
        G = 9.81
        tr = np.deg2rad(theta)
        t_flight = 2 * v0 * np.sin(tr) / G
        t_pts    = np.linspace(0, t_flight, 80)

        x_true = v0 * np.cos(tr) * t_pts
        y_true = v0 * np.sin(tr) * t_pts - 0.5 * G * t_pts ** 2

        X_in  = np.column_stack([np.full(80, v0), np.full(80, theta), t_pts])
        pred  = model.predict(X_in, verbose=0)
        x_pred, y_pred = pred[:, 0], pred[:, 1]

        self._line_traj_t.set_data(x_true, y_true)
        self._line_traj_p.set_data(x_pred, y_pred)
        R_true = float(x_true[-1])
        R_pred = float(x_pred[-1])
        mse    = float(np.mean((pred - np.column_stack([x_true, y_true])) ** 2))
        self._ax_traj.set_title(
            f'v₀={v0:.0f} m/s  θ={theta:.0f}°  R_true={R_true:.1f}m  R_pred={R_pred:.1f}m  MSE={mse:.3f}',
            fontweight='bold', fontsize=9)
        self._ax_traj.relim(); self._ax_traj.autoscale_view()

        self._line_xt_t.set_data(t_pts, x_true); self._line_xt_p.set_data(t_pts, x_pred)
        self._line_yt_t.set_data(t_pts, y_true); self._line_yt_p.set_data(t_pts, y_pred)
        for ax in (self._ax_xt, self._ax_yt):
            ax.relim(); ax.autoscale_view()
        self._ax_xt.set_title('x(t)', fontweight='bold')
        self._ax_yt.set_title('y(t)', fontweight='bold')

        if history:
            full_loss = history.history['loss']
            self._loss_line.set_data(list(range(1, len(full_loss)+1)), full_loss)
            self._ax_xt.relim(); self._ax_xt.autoscale_view()
        self.mpl.draw()

    def _on_reset_impl(self):
        self._epochs_buf.clear(); self._losses_buf.clear()
        for ax in (self._ax_traj, self._ax_xt, self._ax_yt):
            ax.cla()
        self._init_artists()

    def get_param_values(self):
        return {**self._pg_sim.values(), **self._pg_tr.values()}

    def get_metrics(self):
        return {}
