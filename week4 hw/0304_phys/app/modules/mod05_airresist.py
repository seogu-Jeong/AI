"""
MOD-05 — Air Resistance Projectile (RK4 + AI Range Prediction)
FR-MOD05-01..14  |  SRS-PHYSAI-003 §4.5–4.7
"""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QLabel,
                                QProgressBar, QPushButton)
from PySide6.QtGui import QFont

from app.ml.data_generators import DataGenerators
from app.ml.models import ModelFactory
from app.ml.training_worker import TrainingConfig
from app.ml.loading_worker import LoadingWorker
from app.physics.projectile import ProjectilePhysics
from app.widgets.matplotlib_widget import MatplotlibWidget
from app.widgets.param_group import ParamGroup
from .base_module import BaseModule, ModuleState


class AirResistanceModule(BaseModule):
    MODULE_ID   = "MOD-05"
    MODULE_NAME = "Air Resistance Projectile"
    MODULE_DESC = "RK4 simulates drag. NN learns (v₀, θ) → range R. Compares vacuum vs reality."

    def __init__(self, parent=None):
        self._trained_k: float = None   # k value at training time
        super().__init__(parent)

    def _setup_param_panel(self) -> QWidget:
        panel = QWidget(); lay = QVBoxLayout(panel)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)

        self._pg_launch = ParamGroup("Launch Parameters")
        self._v0_sl = self._pg_launch.add_slider('v0', label="Init Velocity v₀",
            min_val=10, max_val=100, default=50, step=1, unit="m/s", decimals=0)
        self._th_sl = self._pg_launch.add_slider('theta', label="Launch Angle θ",
            min_val=10, max_val=80, default=45, step=1, unit="°", decimals=0)
        self._k_sl  = self._pg_launch.add_slider('k', label="Drag Coeff k",
            min_val=0.0, max_val=0.20, default=0.05, step=0.01, decimals=2)
        lay.addWidget(self._pg_launch)

        self._pg_train = ParamGroup("Training Config")
        self._n_sl  = self._pg_train.add_slider('n', label="Train Samples",
            min_val=200, max_val=5000, default=2000, step=100, decimals=0)
        self._ep_sl = self._pg_train.add_slider('epochs', label="Epochs",
            min_val=20, max_val=500, default=100, step=10, decimals=0)
        lay.addWidget(self._pg_train)

        # Data generation progress (FR-MOD05-08)
        self._gen_group = QGroupBox("Data Generation")
        glay = QVBoxLayout(self._gen_group)
        self._gen_bar = QProgressBar(); self._gen_bar.setRange(0, 2000)
        self._gen_lbl = QLabel("Click Run to generate data + train")
        self._gen_lbl.setStyleSheet("font-size:9pt;color:#757575;")
        glay.addWidget(self._gen_bar); glay.addWidget(self._gen_lbl)
        lay.addWidget(self._gen_group)

        # Live info panel (FR-MOD05-14)
        self._info_group = QGroupBox("Physics Info")
        ilay = QVBoxLayout(self._info_group)
        self._vac_lbl  = QLabel("Vacuum Range:   —")
        self._air_lbl  = QLabel("Air Resist Range: —")
        self._ai_lbl   = QLabel("AI Predicted:   —")
        self._red_lbl  = QLabel("Range Reduction: —")
        self._opt_lbl  = QLabel("Optimal Angle:   —")
        bold = QFont(); bold.setBold(True)
        for lbl in (self._vac_lbl, self._air_lbl, self._ai_lbl, self._red_lbl, self._opt_lbl):
            ilay.addWidget(lbl)
        lay.addWidget(self._info_group)

        # FR-MOD05-09: drag coefficient warning
        self._k_warn_lbl = QLabel("")
        self._k_warn_lbl.setWordWrap(True)
        self._k_warn_lbl.setStyleSheet("color:#FB8C00;font-size:9pt;")
        lay.addWidget(self._k_warn_lbl)

        self._v0_sl.value_changed.connect(lambda _: self._on_phys_changed())
        self._th_sl.value_changed.connect(lambda _: self._on_phys_changed())
        self._k_sl.value_changed.connect(lambda _: self._on_k_changed())
        self._pg_train.any_value_changed.connect(lambda *_: self._mark_dirty())
        return panel

    def _setup_plot_area(self) -> QWidget:
        self.mpl = MatplotlibWidget(figsize=(14, 10))
        self._setup_plots()
        return self.mpl

    def _setup_plots(self):
        gs = self.mpl.fresh_gridspec(2, 2, hspace=0.4, wspace=0.35)
        fig = self.mpl.figure
        self._ax_traj  = fig.add_subplot(gs[0, 0])
        self._ax_range = fig.add_subplot(gs[0, 1])
        self._ax_perf  = fig.add_subplot(gs[1, 0])
        self._ax_loss  = fig.add_subplot(gs[1, 1])

        self._ax_traj.set_xlabel('x (m)'); self._ax_traj.set_ylabel('y (m)')
        self._ax_traj.set_title('Trajectory Comparison', fontweight='bold')
        self._ax_traj.grid(True, alpha=0.3)

        self._ax_range.set_xlabel('Angle θ (°)'); self._ax_range.set_ylabel('Range R (m)')
        self._ax_range.set_title('Range vs Launch Angle', fontweight='bold')
        self._ax_range.grid(True, alpha=0.3)

        self._ax_perf.set_xlabel('True Range (m)'); self._ax_perf.set_ylabel('Pred Range (m)')
        self._ax_perf.set_title('AI Performance', fontweight='bold')
        self._ax_perf.grid(True, alpha=0.3)

        self._ax_loss.set_xlabel('Epoch'); self._ax_loss.set_ylabel('Loss (MSE, log)')
        self._ax_loss.set_title('Training Loss', fontweight='bold')
        self._ax_loss.set_yscale('log'); self._ax_loss.grid(True, alpha=0.3, which='both')
        self._line_loss, = self._ax_loss.plot([], [], 'g-', lw=1.8)
        self._epochs_buf = []; self._losses_buf = []
        self.mpl.draw()

    # ── run() override: LoadingWorker first, then TrainingWorker ─────────────
    def run(self):
        if self._state == ModuleState.TRAINING: return
        self._apply_state(ModuleState.TRAINING)
        self._epochs_buf.clear(); self._losses_buf.clear()
        self._line_loss.set_data([], [])
        self._gen_bar.setValue(0)
        k  = self._k_sl.value
        n  = int(self._n_sl.value)
        self._gen_lbl.setText(f"Generating {n} RK4 simulations (k={k:.2f})…")
        self._k_warn_lbl.setText("")

        self._loader = LoadingWorker(n_samples=n, k=k, parent=self)
        self._loader.progress_updated.connect(self._gen_bar.setValue)
        self._loader.data_ready.connect(self._on_data_ready)
        self._loader.loading_error.connect(self._on_training_error)
        self._loader.start()

    def _on_data_ready(self, X: np.ndarray, y: np.ndarray):
        """Called after LoadingWorker finishes; now start TrainingWorker."""
        self._gen_lbl.setText(f"Data ready ({len(X)} samples). Training…")
        self._X_train = X; self._y_train = y
        self._trained_k = self._k_sl.value
        model  = ModelFactory.air_resistance_range()
        config = self._get_training_config()
        self._progress.start(config.epochs)
        import time; self._start_ms = time.monotonic()
        from app.ml.training_worker import TrainingWorker
        self._worker = TrainingWorker(model, X, y, config, parent=self)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.training_finished.connect(self._on_training_finished)
        self._worker.training_error.connect(self._on_training_error)
        self._worker.start()

    def stop(self):
        if hasattr(self, '_loader') and self._loader.isRunning():
            self._loader.request_stop()
        super().stop()

    # ── BaseModule interface ──────────────────────────────────────────────────
    def _build_model(self): return ModelFactory.air_resistance_range()   # unused — run() is overridden
    def _generate_data(self): return (self._X_train, self._y_train)
    def _get_training_config(self) -> TrainingConfig:
        return TrainingConfig(epochs=int(self._ep_sl.value),
                              validation_split=0.2, log_interval=10)

    def _on_progress(self, epoch, loss, val_loss):
        self._progress.update(epoch, self._get_training_config().epochs, loss, val_loss)
        self._epochs_buf.append(epoch); self._losses_buf.append(loss)
        self._line_loss.set_data(self._epochs_buf, self._losses_buf)
        self._ax_loss.relim(); self._ax_loss.autoscale_view()
        self.mpl.draw_idle()

    def _on_training_finished_impl(self, model, history):
        self._render_all(model)
        self._update_info_panel()

    def _on_phys_changed(self):
        """FR-MOD05-14: live info update + trajectory re-sim when TRAINED."""
        if self._state in (ModuleState.TRAINED, ModuleState.DIRTY):
            self._update_info_panel()
            self._render_trajectory_panel(self._model)

    def _on_k_changed(self):
        """FR-MOD05-05,09: re-sim RK4 immediately; warn if trained k differs."""
        if self._state in (ModuleState.TRAINED, ModuleState.DIRTY):
            if self._trained_k is not None and abs(self._k_sl.value - self._trained_k) > 1e-6:
                self._k_warn_lbl.setText(
                    "Drag coefficient changed. Retraining required for accurate AI predictions.")
            else:
                self._k_warn_lbl.setText("")
            self._render_trajectory_panel(self._model)
            self._update_info_panel()
        self._mark_dirty()

    # ── Renderers ─────────────────────────────────────────────────────────────
    def _render_all(self, model):
        self._render_trajectory_panel(model)
        self._render_range_panel(model)
        self._render_performance_panel(model)
        self.mpl.draw()

    def _render_trajectory_panel(self, model):
        """FR-MOD05-10: trajectory + AI star marker."""
        ax = self._ax_traj; ax.clear()
        v0 = self._v0_sl.value; theta = self._th_sl.value; k = self._k_sl.value
        phys_air = ProjectilePhysics(k=k)
        phys_vac = ProjectilePhysics(k=0.0)
        traj_air = phys_air.simulate(v0, theta)
        traj_vac = phys_vac.simulate(v0, theta)
        R_pred   = float(model.predict([[v0, theta]], verbose=0)[0, 0]) if model else 0.0

        ax.plot(traj_vac[:, 0], traj_vac[:, 1], 'b--', lw=2, label='Vacuum (theory)')
        ax.plot(traj_air[:, 0], traj_air[:, 1], 'r-',  lw=2, label='Air Resistance (RK4)')
        ax.scatter([R_pred], [0], c='green', s=200, marker='*', zorder=5,
                   label=f'AI Pred ({R_pred:.1f} m)')

        R_air = phys_air.landing_range(v0, theta)
        R_vac = phys_vac.landing_range(v0, theta)
        red   = (1 - R_air / R_vac) * 100 if R_vac > 0 else 0
        ax.text(0.05, 0.95, f"Range reduction: {red:.1f}%",
                transform=ax.transAxes, va='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_title(f'Trajectory (v₀={v0:.0f}m/s, θ={theta:.0f}°, k={k:.2f})', fontweight='bold', fontsize=9)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    def _render_range_panel(self, model):
        """FR-MOD05-11,12: Range vs Angle for v₀=60 m/s."""
        ax  = self._ax_range; ax.clear()
        angles    = np.linspace(10, 80, 50)
        v0_fixed  = 60.0
        phys_air  = ProjectilePhysics(k=self._k_sl.value)
        phys_vac  = ProjectilePhysics(k=0.0)
        R_vac     = np.array([phys_vac.vacuum_range(v0_fixed, a) for a in angles])
        R_air     = np.array([phys_air.landing_range(v0_fixed, a) for a in angles])
        X_in      = np.column_stack([np.full(50, v0_fixed), angles])
        R_ai      = model.predict(X_in, verbose=0).flatten() if model else np.zeros(50)

        ax.plot(angles, R_vac, 'b--', lw=2, label='Vacuum theory')
        ax.plot(angles, R_air, 'r-',  lw=2, label='RK4 (air resistance)')
        ax.scatter(angles, R_ai, c='green', s=20, label='AI predictions')

        # FR-MOD05-12: optimal angle annotation
        opt_idx   = int(np.argmax(R_air))
        opt_angle = float(angles[opt_idx])
        ax.axvline(opt_angle, color='r', ls='--', lw=1.5, alpha=0.7,
                   label=f'Optimal θ={opt_angle:.1f}°')
        ax.set_xlabel('Angle θ (°)'); ax.set_ylabel('Range R (m)')
        ax.set_title(f'Range vs Angle  (v₀={v0_fixed}m/s, k={self._k_sl.value:.2f})',
                     fontweight='bold', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    def _render_performance_panel(self, model):
        """FR-MOD05-13: True vs Predicted scatter + R²."""
        ax = self._ax_perf; ax.clear()
        # Test set
        rng = np.random.default_rng(seed=42)
        v0t = rng.uniform(10, 100, 200); ath = rng.uniform(10, 80, 200)
        phys = ProjectilePhysics(k=self._trained_k or self._k_sl.value)
        R_true = np.array([phys.landing_range(v, a) for v, a in zip(v0t, ath)])
        R_pred = model.predict(np.column_stack([v0t, ath]), verbose=0).flatten()
        ss_res = np.sum((R_true - R_pred)**2)
        ss_tot = np.sum((R_true - R_true.mean())**2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.scatter(R_true, R_pred, alpha=0.4, s=15, c='C0')
        mn, mx = R_true.min(), R_true.max()
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, label='y=x')
        ax.set_xlabel('True Range (m)'); ax.set_ylabel('Predicted Range (m)')
        ax.set_title(f'AI Performance  R²={r2:.4f}', fontweight='bold', fontsize=9)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    def _update_info_panel(self):
        """FR-MOD05-14: live panel — Vacuum/Air/AI ranges, reduction, optimal angle."""
        v0 = self._v0_sl.value; theta = self._th_sl.value; k = self._k_sl.value
        phys_air = ProjectilePhysics(k=k)
        phys_vac = ProjectilePhysics(k=0.0)
        R_vac    = phys_vac.landing_range(v0, theta)
        R_air    = phys_air.landing_range(v0, theta)
        red      = (1 - R_air / R_vac) * 100 if R_vac > 0 else 0.0

        self._vac_lbl.setText(f"Vacuum Range:    {R_vac:.1f} m")
        self._air_lbl.setText(f"Air Resist Range: {R_air:.1f} m")

        if self._model is not None:
            R_ai = float(self._model.predict([[v0, theta]], verbose=0)[0, 0])
            self._ai_lbl.setText(f"AI Predicted:    {R_ai:.1f} m")
        else:
            self._ai_lbl.setText("AI Predicted:    — (train first)")

        self._red_lbl.setText(f"Range Reduction: {red:.1f}%")

        # Find optimal angle
        angles  = np.linspace(10, 80, 50)
        R_scan  = [phys_air.landing_range(v0, a) for a in angles]
        opt_ang = float(angles[int(np.argmax(R_scan))])
        self._opt_lbl.setText(f"Optimal Angle:   {opt_ang:.1f}°")

    def _on_reset_impl(self):
        self._k_warn_lbl.setText("")
        self._gen_lbl.setText("Click Run to generate data + train")
        self._gen_bar.setValue(0)
        self._epochs_buf.clear(); self._losses_buf.clear()
        self._setup_plots()
        for lbl in (self._vac_lbl, self._air_lbl, self._ai_lbl, self._red_lbl, self._opt_lbl):
            lbl.setText(lbl.text().split(':')[0] + ': —')

    def get_param_values(self):
        return {**self._pg_launch.values(), **self._pg_train.values()}
