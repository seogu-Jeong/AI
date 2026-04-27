"""
MOD-03 — Overfitting / Underfitting Demonstration
Three models trained simultaneously: underfit, good fit, overfit.
"""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QLabel, QHBoxLayout,
)
from PySide6.QtGui import QFont

from app.ml.data_generators import DataGenerators
from app.ml.models import ModelFactory
from app.ml.training_worker import TrainingWorker, TrainingConfig
from app.widgets.matplotlib_widget import MatplotlibWidget
from app.widgets.param_group import ParamGroup
from app.widgets.progress_panel import ProgressPanel
from .base_module import BaseModule, ModuleState

COLORS = {
    'underfit': '#2196F3',
    'good':     '#4CAF50',
    'overfit':  '#F44336',
    'true':     '#212121',
}


class OverfittingDemoModule(BaseModule):
    MODULE_ID   = "MOD-03"
    MODULE_NAME = "Overfitting Demo"
    MODULE_DESC = "Three models (underfit / good / overfit) trained in parallel on noisy data"

    def _setup_param_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._pg = ParamGroup("Dataset & Training")
        self._noise_sl = self._pg.add_slider('noise', label="Noise σ",
            min_val=0.05, max_val=1.0, default=0.3, step=0.05, decimals=2,
            tooltip="Standard deviation of additive Gaussian noise")
        self._n_sl = self._pg.add_slider('n_samples', label="Train Samples",
            min_val=30, max_val=300, default=100, step=10, decimals=0,
            tooltip="Number of noisy training points")
        self._ep_sl = self._pg.add_slider('epochs', label="Epochs",
            min_val=50, max_val=1000, default=200, step=50, decimals=0)
        layout.addWidget(self._pg)

        # Per-model progress
        self._prog_labels: dict = {}
        for key, color in COLORS.items():
            if key == 'true':
                continue
            lbl = QLabel(f"● {key.capitalize()}")
            lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 10pt;")
            layout.addWidget(lbl)
            prog = ProgressPanel()
            layout.addWidget(prog)
            self._prog_labels[key] = prog

        self._pg.any_value_changed.connect(lambda *_: self._mark_dirty())
        return panel

    def _setup_plot_area(self) -> QWidget:
        self._plot_tabs = QTabWidget()
        self.mpl_pred   = MatplotlibWidget(figsize=(12, 5))
        self.mpl_curves = MatplotlibWidget(figsize=(15, 5))
        self.mpl_errors = MatplotlibWidget(figsize=(14, 6))
        self._plot_tabs.addTab(self.mpl_pred,   "📈  Predictions")
        self._plot_tabs.addTab(self.mpl_curves, "📉  Loss Curves")
        self._plot_tabs.addTab(self.mpl_errors, "⚡  Error Analysis")
        self._init_artists()
        return self._plot_tabs

    def _init_artists(self):
        ax = self.mpl_pred.fresh_axes()
        ax.set_title('Model Predictions vs True Function', fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('f(x)')
        ax.grid(True, alpha=0.25)
        self.mpl_pred.draw()

        axes = self.mpl_curves.fresh_axes(1, 3)
        titles = ['Underfit  [4]', 'Good Fit  [32,16]', 'Overfit  [256,128,64,32]']
        for ax, t in zip(axes, titles):
            ax.set_title(t, fontweight='bold', fontsize=9)
            ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.25, which='both')
        self._ax_curves = axes
        self.mpl_curves.draw()

        ax_e = self.mpl_errors.fresh_axes()
        ax_e.set_title('Test Error Distribution', fontweight='bold')
        ax_e.set_xlabel('|error|'); ax_e.set_ylabel('Count')
        ax_e.grid(True, alpha=0.25)
        self._ax_err = ax_e
        self.mpl_errors.draw()

    # ── BaseModule overrides ─────────────────────────────────────────────────
    def _build_model(self):
        return None   # not used; multi-model training below

    def _generate_data(self):
        n = int(self._n_sl.value)
        noise = self._noise_sl.value
        X_tr, y_tr, X_val, y_val, X_te, y_te = DataGenerators.overfitting(
            n_train=n, noise=noise)
        self._X_te, self._y_te = X_te, y_te
        return X_tr, y_tr

    def _get_training_config(self) -> TrainingConfig:
        return TrainingConfig(epochs=int(self._ep_sl.value), validation_split=0.0,
                              use_reduce_lr=False)

    def run(self):
        """Override to start three workers simultaneously."""
        if self._state == ModuleState.TRAINING:
            return
        n     = int(self._n_sl.value)
        noise = self._noise_sl.value
        X_tr, y_tr, X_val, y_val, X_te, y_te = DataGenerators.overfitting(
            n_train=n, noise=noise)
        self._X_te, self._y_te = X_te, y_te
        epochs = int(self._ep_sl.value)

        models = ModelFactory.overfitting_suite()
        config = TrainingConfig(epochs=epochs, validation_split=0.2,
                                use_reduce_lr=False, log_interval=10)

        self._workers: dict = {}
        self._histories: dict = {}
        self._finished: set = set()
        self._trained_models: dict = {}

        for key in ('underfit', 'good', 'overfit'):
            w = TrainingWorker(models[key], X_tr, y_tr, config, parent=self)
            w.progress_updated.connect(
                lambda e, l, vl, k=key: self._on_model_progress(k, e, l, vl))
            w.training_finished.connect(
                lambda m, h, k=key: self._on_model_finished(k, m, h))
            w.training_error.connect(
                lambda msg, k=key: self._on_training_error(msg))
            self._workers[key] = w
            self._prog_labels[key].start(epochs)

        self._apply_state(ModuleState.TRAINING)
        for w in self._workers.values():
            w.start()

    def stop(self):
        for w in getattr(self, '_workers', {}).values():
            if w.isRunning():
                w.request_stop()
        self._apply_state(ModuleState.IDLE)

    def _on_model_progress(self, key, epoch, loss, val_loss):
        self._prog_labels[key].update(epoch, int(self._ep_sl.value), loss, val_loss)

    def _on_model_finished(self, key, model, history):
        self._trained_models[key] = model
        self._histories[key]      = history
        self._finished.add(key)
        self._prog_labels[key].complete()

        if self._finished == {'underfit', 'good', 'overfit'}:
            self._render_all()
            self._apply_state(ModuleState.TRAINED)

    def _render_all(self):
        X_te = self._X_te
        y_te = self._y_te.flatten()
        x_flat = X_te.flatten()

        # ── Predictions tab ───────────────────────────────────────────────────
        ax = self.mpl_pred.fresh_axes()
        ax.scatter(x_flat, y_te, c='#9E9E9E', s=10, alpha=0.5, label='True function', zorder=1)
        for key, color in [('underfit', COLORS['underfit']),
                            ('good',     COLORS['good']),
                            ('overfit',  COLORS['overfit'])]:
            m    = self._trained_models[key]
            pred = m.predict(X_te, verbose=0).flatten()
            ax.plot(x_flat, pred, '-', color=color, lw=2.0, label=key.capitalize(), zorder=3)
        ax.set_title('Predictions vs True: f(x) = sin(2x) + 0.5x', fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('f(x)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
        self.mpl_pred.draw()

        # ── Loss curves tab ───────────────────────────────────────────────────
        axes = self.mpl_curves.fresh_axes(1, 3)
        titles = ['Underfit  [4 units]', 'Good Fit  [32→16 + Dropout]', 'Overfit  [256→128→64→32]']
        keys   = ['underfit', 'good', 'overfit']
        colors = [COLORS['underfit'], COLORS['good'], COLORS['overfit']]
        for ax, t, k, c in zip(axes, titles, keys, colors):
            h     = self._histories[k].history
            loss  = h.get('loss', [])
            v_loss = h.get('val_loss', [])
            ep    = list(range(1, len(loss) + 1))
            ax.plot(ep, loss,  '-',  color=c, lw=2.0, label='Train')
            if v_loss:
                ax.plot(ep, v_loss, '--', color=c, lw=1.5, alpha=0.7, label='Val', linestyle='--')
            ax.set_title(f'{t}\nFinal: {loss[-1]:.4f}', fontweight='bold', fontsize=9)
            ax.set_xlabel('Epoch'); ax.set_ylabel('MSE'); ax.set_yscale('log')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.25, which='both')
        self.mpl_curves.draw()

        # ── Error analysis tab ────────────────────────────────────────────────
        ax_e = self.mpl_errors.fresh_axes()
        for key, color in [('underfit', COLORS['underfit']),
                            ('good',     COLORS['good']),
                            ('overfit',  COLORS['overfit'])]:
            m    = self._trained_models[key]
            pred = m.predict(X_te, verbose=0).flatten()
            err  = np.abs(pred - y_te)
            ax_e.hist(err, bins=30, alpha=0.55, color=color, label=f'{key.capitalize()} (MAE={err.mean():.3f})')
        ax_e.set_title('Test Absolute Error Distribution', fontweight='bold')
        ax_e.set_xlabel('|error|'); ax_e.set_ylabel('Count')
        ax_e.legend(fontsize=9); ax_e.grid(True, alpha=0.25)
        self.mpl_errors.draw()

    def _on_training_finished_impl(self, model, history):
        pass  # handled in _on_model_finished

    def _on_reset_impl(self):
        self._init_artists()
        for prog in self._prog_labels.values():
            prog.reset()

    def get_param_values(self):
        return self._pg.values()

    def get_metrics(self):
        metrics = {}
        for key in ('underfit', 'good', 'overfit'):
            if hasattr(self, '_trained_models') and key in self._trained_models:
                m    = self._trained_models[key]
                pred = m.predict(self._X_te, verbose=0).flatten()
                metrics[f'{key}_mae'] = float(np.mean(np.abs(pred - self._y_te.flatten())))
        return metrics
