"""
MOD-05 — Air Resistance Projectile
Compares vacuum vs. air resistance trajectories; NN predicts landing range.
"""
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout

from app.physics.projectile import ProjectilePhysics
from app.ml.data_generators import DataGenerators
from app.ml.models import ModelFactory
from app.ml.training_worker import TrainingConfig
from app.widgets.matplotlib_widget import MatplotlibWidget
from app.widgets.param_group import ParamGroup
from .base_module import BaseModule


class AirResistanceModule(BaseModule):
    MODULE_ID   = "MOD-05"
    MODULE_NAME = "Air Resistance"
    MODULE_DESC = "NN predicts landing range with drag. Compares vacuum vs. air resistance via RK4"

    def _setup_param_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._pg_sim = ParamGroup("Projectile Parameters")
        self._v0_sl = self._pg_sim.add_slider('v0', label="Initial Speed v₀",
            min_val=10, max_val=100, default=40, step=5, unit="m/s", decimals=0,
            tooltip="Initial projectile speed")
        self._th_sl = self._pg_sim.add_slider('theta', label="Launch Angle θ",
            min_val=15, max_val=75, default=45, step=5, unit="°", decimals=0,
            tooltip="Launch angle above horizontal")
        self._k_sl  = self._pg_sim.add_slider('k', label="Drag Coefficient k",
            min_val=0.0, max_val=0.2, default=0.05, step=0.01, unit="kg⁻¹", decimals=3,
            tooltip="k = C_d·A·ρ/(2m). 0 = vacuum, 0.05 = typical ball")
        layout.addWidget(self._pg_sim)

        self._pg_tr = ParamGroup("Training")
        self._ep_sl = self._pg_tr.add_slider('epochs', label="Epochs",
            min_val=50, max_val=500, default=200, step=50, decimals=0)
        self._n_sl  = self._pg_tr.add_slider('n_samples', label="Samples",
            min_val=200, max_val=2000, default=800, step=200, decimals=0,
            tooltip="Fewer samples → faster data gen (RK4 per sample)")
        layout.addWidget(self._pg_tr)

        for pg in (self._pg_sim, self._pg_tr):
            pg.any_value_changed.connect(lambda *_: self._mark_dirty())
        return panel

    def _setup_plot_area(self) -> QWidget:
        self.mpl = MatplotlibWidget(figsize=(14, 10))
        self._epochs_buf, self._losses_buf = [], []
        gs = self.mpl.fresh_gridspec(2, 2, hspace=0.45, wspace=0.38)
        fig = self.mpl.figure
        self._ax_traj  = fig.add_subplot(gs[0, 0])
        self._ax_range = fig.add_subplot(gs[0, 1])
        self._ax_sweep = fig.add_subplot(gs[1, 0])
        self._ax_loss  = fig.add_subplot(gs[1, 1])
        self._annotate_axes()
        self.mpl.draw()
        return self.mpl

    def _annotate_axes(self):
        for ax, title, xlabel, ylabel in [
            (self._ax_traj,  'Trajectory Comparison',         'x (m)',       'y (m)'),
            (self._ax_range, 'Range vs Launch Angle',         'θ (°)',       'Range (m)'),
            (self._ax_sweep, 'Range vs Initial Speed',        'v₀ (m/s)',    'Range (m)'),
            (self._ax_loss,  'Training Loss',                  'Epoch',       'MSE (log)'),
        ]:
            ax.set_title(title, fontweight='bold', fontsize=9)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
        self._ax_loss.set_yscale('log')
        self._ax_loss.grid(True, alpha=0.25, which='both')

    # ── BaseModule interface ─────────────────────────────────────────────────
    def _build_model(self):
        return ModelFactory.air_resistance_range()

    def _generate_data(self):
        k = self._k_sl.value
        n = int(self._n_sl.value)

        def prog_cb(i):
            pct = int(100 * i / n)
            self._progress._bar.setValue(pct)

        X, y = DataGenerators.air_resistance(n_samples=n, k=k, progress_cb=prog_cb)
        self._k_used = k
        return X, y

    def _get_training_config(self) -> TrainingConfig:
        return TrainingConfig(epochs=int(self._ep_sl.value), batch_size=64)

    def _on_progress(self, epoch: int, loss: float, val_loss: float):
        super()._on_progress(epoch, loss, val_loss)
        cfg = self._get_training_config()
        self._epochs_buf.append(epoch)
        self._losses_buf.append(loss)
        # Update loss subplot
        epochs = self._epochs_buf
        self._ax_loss.cla()
        self._ax_loss.plot(epochs, self._losses_buf, 'C2-', lw=1.8)
        self._ax_loss.set_title(f'Loss {loss:.6f}', fontweight='bold', fontsize=9)
        self._ax_loss.set_xlabel('Epoch'); self._ax_loss.set_ylabel('MSE')
        self._ax_loss.set_yscale('log'); self._ax_loss.grid(True, alpha=0.25, which='both')
        self.mpl.draw_idle()

    def _on_training_finished_impl(self, model, history):
        v0    = self._v0_sl.value
        theta = self._th_sl.value
        k     = self._k_used

        # ── Trajectory panel ──────────────────────────────────────────────────
        p_air = ProjectilePhysics(k=k)
        p_vac = ProjectilePhysics(k=0.0)
        t_air = p_air.simulate(v0, theta)
        t_vac = p_vac.simulate(v0, theta)
        R_air = float(t_air[-1, 0])
        R_vac = float(t_vac[-1, 0])
        R_pred = float(model.predict([[v0, theta]], verbose=0)[0, 0])
        reduction = (1 - R_air / R_vac) * 100

        self._ax_traj.cla()
        self._ax_traj.plot(t_vac[:, 0], t_vac[:, 1], 'C0--', lw=2.0, label=f'Vacuum ({R_vac:.1f} m)')
        self._ax_traj.plot(t_air[:, 0], t_air[:, 1], 'C3-',  lw=2.0, label=f'Air (k={k}, {R_air:.1f} m)')
        self._ax_traj.scatter([R_pred], [0], c='C2', s=180, marker='*', zorder=5,
                               label=f'AI Pred ({R_pred:.1f} m)')
        self._ax_traj.set_xlim(left=0); self._ax_traj.set_ylim(bottom=0)
        self._ax_traj.set_title(
            f'v₀={v0:.0f} m/s  θ={theta:.0f}°  k={k:.3f}\nRange reduction: {reduction:.1f}%',
            fontweight='bold', fontsize=9)
        self._ax_traj.set_xlabel('x (m)'); self._ax_traj.set_ylabel('y (m)')
        self._ax_traj.legend(fontsize=8); self._ax_traj.grid(True, alpha=0.25)

        # ── Range vs angle ────────────────────────────────────────────────────
        angles = np.linspace(10, 80, 60)
        R_vac_curve  = np.array([p_vac.vacuum_range(v0, a) for a in angles])
        R_air_curve  = np.array([p_air.landing_range(v0, a) for a in angles])
        X_in = np.column_stack([np.full(60, v0), angles])
        R_nn = model.predict(X_in, verbose=0).flatten()

        self._ax_range.cla()
        self._ax_range.plot(angles, R_vac_curve, 'C0--', lw=2.0, label='Vacuum')
        self._ax_range.plot(angles, R_air_curve, 'C3-',  lw=2.0, label=f'Air (k={k})')
        self._ax_range.plot(angles, R_nn,        'C2:',  lw=2.0, label='NN Pred')
        self._ax_range.set_title(f'Range vs θ  (v₀={v0:.0f} m/s)', fontweight='bold', fontsize=9)
        self._ax_range.set_xlabel('θ (°)'); self._ax_range.set_ylabel('Range (m)')
        self._ax_range.legend(fontsize=8); self._ax_range.grid(True, alpha=0.25)

        # ── Range vs speed ────────────────────────────────────────────────────
        speeds = np.linspace(10, 100, 60)
        R_vac_s = np.array([p_vac.vacuum_range(v, theta) for v in speeds])
        R_air_s = np.array([p_air.landing_range(v, theta) for v in speeds])
        X_in2   = np.column_stack([speeds, np.full(60, theta)])
        R_nn2   = model.predict(X_in2, verbose=0).flatten()

        self._ax_sweep.cla()
        self._ax_sweep.plot(speeds, R_vac_s, 'C0--', lw=2.0, label='Vacuum')
        self._ax_sweep.plot(speeds, R_air_s, 'C3-',  lw=2.0, label=f'Air (k={k})')
        self._ax_sweep.plot(speeds, R_nn2,   'C2:',  lw=2.0, label='NN Pred')
        self._ax_sweep.set_title(f'Range vs v₀  (θ={theta:.0f}°)', fontweight='bold', fontsize=9)
        self._ax_sweep.set_xlabel('v₀ (m/s)'); self._ax_sweep.set_ylabel('Range (m)')
        self._ax_sweep.legend(fontsize=8); self._ax_sweep.grid(True, alpha=0.25)

        # ── Loss (final) ──────────────────────────────────────────────────────
        full_loss = history.history['loss']
        self._ax_loss.cla()
        self._ax_loss.plot(range(1, len(full_loss)+1), full_loss, 'C2-', lw=1.8)
        self._ax_loss.set_title(f'Loss (final: {full_loss[-1]:.2e})', fontweight='bold', fontsize=9)
        self._ax_loss.set_xlabel('Epoch'); self._ax_loss.set_ylabel('MSE')
        self._ax_loss.set_yscale('log'); self._ax_loss.grid(True, alpha=0.25, which='both')

        self.mpl.draw()

    def _on_reset_impl(self):
        self._epochs_buf.clear(); self._losses_buf.clear()
        for ax in (self._ax_traj, self._ax_range, self._ax_sweep, self._ax_loss):
            ax.cla()
        self._annotate_axes()
        self.mpl.draw()

    def get_param_values(self):
        return {**self._pg_sim.values(), **self._pg_tr.values()}

    def get_metrics(self):
        return {}
