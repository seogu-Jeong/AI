"""
MOD-03 — Overfitting / Underfitting Demonstration
SRS-PHYSAI-002 §4.5–4.6  (FR-MOD03-01 through FR-MOD03-10)
"""
import numpy as np
from typing import Optional, Dict
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QFrame,
)
from PySide6.QtGui import QFont

from app.ml.data_generators import DataGenerators
from app.ml.models import ModelFactory
from app.ml.training_worker import TrainingWorker, TrainingConfig
from app.widgets.matplotlib_widget import MatplotlibWidget
from app.widgets.param_group import ParamGroup
from app.widgets.progress_panel import ProgressPanel
from .base_module import BaseModule, ModuleState

# FR-MOD03-04: fixed colors
COLORS = {'underfit': '#2196F3', 'good': '#4CAF50', 'overfit': '#F44336'}
LABELS = {'underfit': 'Underfit', 'good': 'Good Fit', 'overfit': 'Overfit'}


def _detect_overfit_epoch(train_losses: list, val_losses: list, warmup: int = 20) -> Optional[int]:
    """FR-MOD03-05: first epoch where val_loss > 2 × train_loss after warmup."""
    for i in range(warmup, min(len(train_losses), len(val_losses))):
        if val_losses[i] > 2.0 * train_losses[i]:
            return i
    return None


class _ModelStatusRow(QWidget):
    """FR-MOD03-03: per-model status indicator ⏳/✓/✗ + ProgressPanel."""
    def __init__(self, key: str, color: str, parent=None):
        super().__init__(parent)
        self._key = key
        lay = QVBoxLayout(self); lay.setContentsMargins(0,4,0,4); lay.setSpacing(4)

        header = QHBoxLayout()
        dot = QLabel("●"); dot.setStyleSheet(f"color:{color};font-size:14pt;")
        self._name_lbl = QLabel(f" {LABELS[key]}")
        f = QFont(); f.setBold(True); f.setPointSize(10); self._name_lbl.setFont(f)
        self._status_lbl = QLabel("⏳")
        self._status_lbl.setStyleSheet("font-size:13pt;")
        header.addWidget(dot); header.addWidget(self._name_lbl)
        header.addStretch(); header.addWidget(self._status_lbl)
        lay.addLayout(header)

        self.progress = ProgressPanel(); lay.addWidget(self.progress)

        # FR-MOD03-09: overfitting badge (hidden initially)
        self._badge = QLabel()
        self._badge.setWordWrap(True)
        self._badge.setStyleSheet("color:#FF9800;font-weight:bold;font-size:9pt;padding:3px 6px;"
                                   "background:#FFF3E0;border-radius:4px;")
        # FR-MOD03-10: tooltip
        self._badge.setToolTip("Validation loss exceeded 2× training loss.\n"
                                "The model is memorising training noise rather than learning the pattern.")
        self._badge.hide()
        lay.addWidget(self._badge)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#E0E0E0;"); lay.addWidget(sep)

    def set_training(self):
        self._status_lbl.setText("⏳"); self._badge.hide()

    def set_done(self):
        self._status_lbl.setText("✓")
        self._status_lbl.setStyleSheet("font-size:13pt;color:#43A047;")

    def set_error(self):
        self._status_lbl.setText("✗")
        self._status_lbl.setStyleSheet("font-size:13pt;color:#E53935;")

    def show_overfit_badge(self, epoch: int):
        """FR-MOD03-09: show amber badge with epoch number."""
        self._badge.setText(f"⚠ Overfitting Detected at Epoch {epoch}")
        self._badge.show()


class OverfittingDemoModule(BaseModule):
    MODULE_ID   = "MOD-03"
    MODULE_NAME = "Overfitting Demo"
    MODULE_DESC = "Three models (underfit / good / overfit) trained in parallel on identical noisy data"

    # ── parameter panel ───────────────────────────────────────────────────────
    def _setup_param_panel(self) -> QWidget:
        panel = QWidget(); lay = QVBoxLayout(panel)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)

        self._pg = ParamGroup("Dataset & Training")
        self._noise_sl = self._pg.add_slider('noise', label="Noise σ",
            min_val=0.05, max_val=1.0, default=0.3, step=0.05, decimals=2,
            tooltip="Gaussian noise std on training and val data. Test data is always noise-free.")
        self._n_sl = self._pg.add_slider('n_train', label="Train Samples",
            min_val=30, max_val=300, default=100, step=10, decimals=0,
            tooltip="Training set size. Fewer samples → easier to overfit.")
        self._ep_sl = self._pg.add_slider('epochs', label="Epochs",
            min_val=50, max_val=500, default=200, step=50, decimals=0)
        lay.addWidget(self._pg)

        # FR-MOD03-07: change advisory label
        self._rerun_lbl = QLabel("")
        self._rerun_lbl.setWordWrap(True)
        self._rerun_lbl.setStyleSheet("color:#FB8C00;font-size:9pt;padding:4px;")
        lay.addWidget(self._rerun_lbl)

        # Per-model status rows (FR-MOD03-03)
        self._rows: Dict[str, _ModelStatusRow] = {}
        for key, color in COLORS.items():
            row = _ModelStatusRow(key, color); lay.addWidget(row)
            self._rows[key] = row

        # Wire dirty / advisory
        self._pg.any_value_changed.connect(self._on_param_changed)
        return panel

    def _on_param_changed(self, name, value):
        """FR-MOD03-07: show advisory when params change post-training."""
        self._mark_dirty()
        if self._state == ModuleState.DIRTY:
            self._rerun_lbl.setText("Parameters changed. Click Run to retrain with new settings.")
        else:
            self._rerun_lbl.setText("")

    # ── plot area ─────────────────────────────────────────────────────────────
    def _setup_plot_area(self) -> QWidget:
        self._plot_tabs = QTabWidget()
        self.mpl_pred   = MatplotlibWidget(figsize=(12, 5))
        self.mpl_curves = MatplotlibWidget(figsize=(15, 5))
        self.mpl_errors = MatplotlibWidget(figsize=(14, 7))
        self._plot_tabs.addTab(self.mpl_pred,   "📈  Predictions")
        self._plot_tabs.addTab(self.mpl_curves, "📉  Loss Curves")
        self._plot_tabs.addTab(self.mpl_errors, "⚡  Error Analysis")
        self._init_empty_plots()
        return self._plot_tabs

    def _init_empty_plots(self):
        ax = self.mpl_pred.fresh_axes()
        ax.set_title('Model Predictions (run to train)', fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('f(x)'); ax.grid(True, alpha=0.25)
        self.mpl_pred.draw()
        for mpl in (self.mpl_curves, self.mpl_errors):
            a = mpl.fresh_axes(); a.set_visible(False); mpl.draw()

    # ── training orchestration ────────────────────────────────────────────────
    # FR-MOD03-02: three threads started in tight loop
    def run(self):
        if self._state == ModuleState.TRAINING: return
        n     = int(self._n_sl.value)
        noise = self._noise_sl.value
        epochs = int(self._ep_sl.value)

        X_tr, y_tr, X_val, y_val, X_te, y_te = DataGenerators.overfitting(
            n_train=n, n_val=50, noise=noise)
        self._X_te = X_te; self._y_te = y_te

        models  = ModelFactory.overfitting_suite()
        config  = TrainingConfig(epochs=epochs, log_interval=10,
                                 use_reduce_lr=False, validation_split=0.0)

        self._workers:  Dict[str, TrainingWorker] = {}
        self._histories: Dict = {}
        self._trained:   Dict = {}
        self._finished:  set  = set()
        self._all_train_losses: Dict[str, list] = {}
        self._all_val_losses:   Dict[str, list] = {}

        for key in ('underfit', 'good', 'overfit'):
            self._rows[key].set_training()
            self._rows[key].progress.start(epochs)
            self._all_train_losses[key] = []
            self._all_val_losses[key]   = []
            w = TrainingWorker(models[key], X_tr, y_tr, config,
                               X_val=X_val, y_val=y_val, parent=self)
            w.progress_updated.connect(
                lambda e, l, vl, k=key: self._on_model_progress(k, e, l, vl))
            w.training_finished.connect(
                lambda m, h, k=key: self._on_model_finished(k, m, h))
            w.training_error.connect(
                lambda msg, k=key: self._on_model_error(k, msg))
            self._workers[key] = w

        self._apply_state(ModuleState.TRAINING)
        self._rerun_lbl.setText("")
        for w in self._workers.values(): w.start()   # FR-MOD03-02: tight loop

    def stop(self):
        for w in getattr(self, '_workers', {}).values():
            if w.isRunning(): w.request_stop()
        self._apply_state(ModuleState.IDLE)

    # ── per-model progress/completion slots ───────────────────────────────────
    def _on_model_progress(self, key: str, epoch: int, loss: float, val_loss: float):
        self._rows[key].progress.update(epoch, int(self._ep_sl.value), loss, val_loss)
        self._all_train_losses[key].append(loss)
        self._all_val_losses[key].append(val_loss)

    def _on_model_finished(self, key: str, model, history):
        self._trained[key]   = model
        self._histories[key] = history
        self._finished.add(key)
        self._rows[key].set_done()
        self._rows[key].progress.complete()

        # FR-MOD03-05: detect overfitting for this model
        if key == 'overfit':
            train_l = history.history.get('loss', [])
            val_l   = history.history.get('val_loss', [])
            of_ep   = _detect_overfit_epoch(train_l, val_l)
            if of_ep is not None:
                self._rows[key].show_overfit_badge(of_ep)   # FR-MOD03-09
            self._overfit_epoch = of_ep

        if self._finished == {'underfit', 'good', 'overfit'}:
            self._render_all_results()
            self._apply_state(ModuleState.TRAINED)
            self._status_lbl.setText("All 3 models trained")

    def _on_model_error(self, key: str, msg: str):
        self._rows[key].set_error()
        self._rows[key].progress.error(msg)

    # ── render all results after all 3 done ───────────────────────────────────
    def _render_all_results(self):
        X_te = self._X_te; y_te = self._y_te.flatten()
        x_flat = X_te.flatten()

        # ── Tab 0: Predictions ────────────────────────────────────────────────
        ax = self.mpl_pred.fresh_axes()
        ax.scatter(x_flat, y_te, c='#9E9E9E', s=10, alpha=0.4, label='True function (noise-free)', zorder=1)
        f = lambda x: np.sin(2*x) + 0.5*x
        ax.plot(x_flat, f(X_te).flatten(), 'k-', lw=2.5, label='True f(x)', zorder=3, alpha=0.85)
        for key in ('underfit', 'good', 'overfit'):
            pred = self._trained[key].predict(X_te, verbose=0).flatten()
            mse  = float(np.mean((pred - y_te)**2))
            # FR-MOD03-08: MSE in legend label
            lbl  = f"{LABELS[key]} (MSE={mse:.4f})"
            ax.plot(x_flat, pred, '-', color=COLORS[key], lw=2.0, label=lbl, zorder=4)
        ax.set_title('f(x) = sin(2x) + 0.5x  — Three Models', fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('f(x)')
        ax.legend(fontsize=9, loc='upper left'); ax.grid(True, alpha=0.25)
        self.mpl_pred.draw()

        # ── Tab 1: Loss Curves ────────────────────────────────────────────────
        axes = self.mpl_curves.fresh_axes(1, 3)
        arch_descs = {'underfit': 'Dense(4)', 'good': 'Dense(32,16)+DO', 'overfit': 'Dense(256,128,64,32)'}
        for ax, key in zip(axes, ('underfit', 'good', 'overfit')):
            h     = self._histories[key].history
            train = h.get('loss', [])
            val   = h.get('val_loss', [])
            ep    = list(range(1, len(train)+1))
            ax.plot(ep, train, '-',  color=COLORS[key], lw=2.0, label='Train')
            if val:
                ax.plot(ep, val, '--', color=COLORS[key], lw=1.5, alpha=0.7, label='Val')
            # FR-MOD03-05: orange vertical line at overfit onset for overfit model
            if key == 'overfit':
                of_ep = getattr(self, '_overfit_epoch', None)
                if of_ep is not None:
                    ax.axvline(of_ep, color='#FF9800', ls='--', lw=2.0,
                               label=f'Overfit onset (ep {of_ep})')
            ax.set_title(f'{LABELS[key]}\n[{arch_descs[key]}]  final:{train[-1]:.4f}',
                         fontweight='bold', fontsize=9)
            ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
            ax.set_yscale('log'); ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25, which='both')
        self.mpl_curves.draw()

        # ── Tab 2: Error Analysis ─────────────────────────────────────────────
        fig = self.mpl_errors.figure; fig.clear()
        gs  = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.38)
        # Top row: histograms per model
        for col, key in enumerate(('underfit', 'good', 'overfit')):
            ax = fig.add_subplot(gs[0, col])
            pred = self._trained[key].predict(X_te, verbose=0).flatten()
            err  = np.abs(pred - y_te)
            ax.hist(err, bins=30, color=COLORS[key], alpha=0.65, edgecolor='none')
            ax.set_title(f'{LABELS[key]}  MAE={err.mean():.4f}', fontweight='bold', fontsize=9)
            ax.set_xlabel('|error|'); ax.set_ylabel('Count'); ax.grid(True, alpha=0.25)

        # Bottom: FR-MOD03-06 performance table
        ax_tbl = fig.add_subplot(gs[1, :])
        ax_tbl.axis('off')
        rows_data = []
        for key in ('underfit', 'good', 'overfit'):
            h     = self._histories[key].history
            pred  = self._trained[key].predict(X_te, verbose=0).flatten()
            tr_l  = h['loss'][-1]
            vl_l  = h.get('val_loss', [tr_l])[-1]
            te_mse = float(np.mean((pred - y_te)**2))
            te_mae = float(np.mean(np.abs(pred - y_te)))
            rows_data.append([LABELS[key], f'{tr_l:.6f}', f'{vl_l:.6f}',
                               f'{te_mse:.6f}', f'{te_mae:.6f}'])
        col_labels = ['Model', 'Train Loss', 'Val Loss', 'Test MSE', 'Test MAE']
        tbl = ax_tbl.table(cellText=rows_data, colLabels=col_labels,
                            loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(10)
        tbl.auto_set_column_width(col=list(range(len(col_labels))))
        # FR-MOD03-06: highlight Good Fit row in #C8E6C9
        good_row_idx = 1   # 0=header, 1=underfit, 2=good, 3=overfit
        for col_idx in range(len(col_labels)):
            tbl[good_row_idx + 1, col_idx].set_facecolor('#C8E6C9')
        ax_tbl.set_title('Performance Summary', fontweight='bold', pad=12)
        self.mpl_errors.draw()

    # ── BaseModule abstract stubs (not used — run() is fully overridden) ──────
    def _build_model(self): return None
    def _generate_data(self): return (None, None)
    def _get_training_config(self): return TrainingConfig()
    def _on_training_finished_impl(self, model, history): pass

    def _on_reset_impl(self):
        for row in self._rows.values():
            row.set_training(); row.progress.reset()
            row._badge.hide()
            row._status_lbl.setText("⏳")
            row._status_lbl.setStyleSheet("font-size:13pt;")
        self._rerun_lbl.setText("")
        self._init_empty_plots()

    def get_param_values(self): return self._pg.values()

    def get_metrics(self):
        metrics = {}
        for key in ('underfit','good','overfit'):
            if hasattr(self,'_trained') and key in self._trained:
                pred = self._trained[key].predict(self._X_te, verbose=0).flatten()
                metrics[f'{key}_mae'] = float(np.mean(np.abs(pred - self._y_te.flatten())))
        return metrics
