"""
MOD-01 — 1D Function Approximation
SRS-PHYSAI-002 §4.1–4.4  (FR-MOD01-01 through FR-MOD01-15)
"""
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from app.ml.data_generators import DataGenerators
from app.ml.models import ModelFactory
from app.ml.training_worker import TrainingConfig
from app.widgets.matplotlib_widget import MatplotlibWidget
from app.widgets.param_group import ParamGroup
from .base_module import BaseModule


# FR-MOD01-01: four presets
ARCH_OPTIONS = list(ModelFactory.ARCH_OPTIONS.keys())   # Small / Medium / Large / XLarge
F04_ARCH     = 'XLarge [128, 128, 64]'                  # FR-MOD01-02
FUNC_OPTIONS = list(DataGenerators.FUNC_MAP.keys())


class FunctionApproximationModule(BaseModule):
    MODULE_ID   = "MOD-01"
    MODULE_NAME = "1D Function Approximation"
    MODULE_DESC = "Neural network regresses mathematical functions (Universal Approximation Theorem)"

    # ── parameter panel ───────────────────────────────────────────────────────
    def _setup_param_panel(self) -> QWidget:
        panel = QWidget(); lay = QVBoxLayout(panel)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)

        # Function group
        self._pg_func = ParamGroup("Target Function")
        self._func_cb = self._pg_func.add_combo(
            'func', "Function:", FUNC_OPTIONS,
            tooltip="Select target function. F-04 requires XLarge architecture.")
        lay.addWidget(self._pg_func)

        # Architecture group
        self._pg_arch = ParamGroup("Network Architecture")
        self._arch_cb = self._pg_arch.add_combo(
            'arch', "Architecture:", ARCH_OPTIONS, default_idx=2,
            tooltip="Hidden-layer configuration. Larger = more capacity.")
        # FR-MOD01-03: parameter count label
        self._param_count_lbl = self._pg_arch.add_label('param_count', "Parameters: —")
        self._param_count_lbl.setStyleSheet("color:#757575;font-size:9pt;")
        self._act_cb = self._pg_arch.add_combo(
            'activation', "Activation:",
            ['tanh', 'relu', 'sigmoid'], default_idx=0,
            # FR-MOD01-07: tooltip explaining tanh preference
            tooltip="tanh: bounded ±1, smooth, ideal for periodic targets (default).\n"
                    "relu: unbounded, risk of dead neurons on zero-crossing functions.\n"
                    "sigmoid: bounded (0,1), asymmetric — not recommended for zero-mean functions.")
        lay.addWidget(self._pg_arch)

        # Training group
        self._pg_train = ParamGroup("Training")
        self._ep_sl = self._pg_train.add_slider('epochs', label="Epochs",
            min_val=100, max_val=8000, default=3000, step=100, decimals=0,
            tooltip="Gradient steps. F-04 benefits from 5000–8000 epochs.")
        self._lr_sl = self._pg_train.add_slider('lr', label="Learning Rate",
            min_val=0.001, max_val=0.05, default=0.01, step=0.001, decimals=3,
            tooltip="Adam initial LR. ReduceLROnPlateau decreases it automatically.")
        lay.addWidget(self._pg_train)

        # Wire dirty state + auto-XLarge + param count
        self._func_cb.currentTextChanged.connect(self._on_function_changed)
        self._arch_cb.currentTextChanged.connect(self._on_arch_changed)
        self._act_cb.currentTextChanged.connect(lambda _: self._mark_dirty())
        self._pg_train.any_value_changed.connect(lambda *_: self._mark_dirty())

        # Initial param count
        self._update_param_count()
        return panel

    def _on_function_changed(self, func_name: str):
        """FR-MOD01-02: auto-select XLarge for F-04; FR-MOD01-05 reset to tanh."""
        if func_name == 'extreme':
            self._arch_cb.blockSignals(True)
            self._arch_cb.setCurrentText(F04_ARCH)
            self._arch_cb.blockSignals(False)
            self._arch_cb.setToolTip("Extreme function requires XLarge architecture.")   # FR-MOD01-02
        else:
            self._arch_cb.setToolTip("")
        self._update_param_count()
        self._mark_dirty()

    def _on_arch_changed(self, _):
        self._update_param_count()
        self._mark_dirty()

    def _update_param_count(self):
        """FR-MOD01-03: display 'Parameters: ~{n:,}'."""
        try:
            arch = ModelFactory.ARCH_OPTIONS[self._arch_cb.currentText()]
            act  = self._act_cb.currentText()
            n    = ModelFactory.count_params(arch, act)
            self._param_count_lbl.setText(f"Parameters: ~{n:,}")
        except Exception:
            self._param_count_lbl.setText("Parameters: —")

    # ── plot area ─────────────────────────────────────────────────────────────
    def _setup_plot_area(self) -> QWidget:
        self.mpl = MatplotlibWidget(figsize=(16, 5))
        self._epochs_buf: list = []; self._losses_buf: list = []
        self._is_f04 = False
        self._init_3panel()
        return self.mpl

    def _init_3panel(self):
        """Standard 3-panel layout for F-01 through F-03."""
        axes = self.mpl.fresh_axes(1, 3)
        self._ax_approx, self._ax_loss, self._ax_err = axes
        self._ax_approx.set_title('Function Approximation', fontweight='bold')
        self._ax_approx.set_xlabel('x'); self._ax_approx.set_ylabel('f(x)')
        self._ax_approx.grid(True, alpha=0.25)
        self._ax_loss.set_title('Training Loss', fontweight='bold')
        self._ax_loss.set_xlabel('Epoch'); self._ax_loss.set_ylabel('MSE (log)')
        self._ax_loss.set_yscale('log'); self._ax_loss.grid(True, alpha=0.25, which='both')
        self._ax_err.set_title('Absolute Error', fontweight='bold')
        self._ax_err.set_xlabel('x'); self._ax_err.set_ylabel('|error|')
        self._ax_err.grid(True, alpha=0.25)
        # Pre-create artists for incremental update (FR-MOD01-09)
        self._line_true,  = self._ax_approx.plot([], [], 'C0-',  lw=2.5, alpha=0.7, label='True')
        self._line_pred,  = self._ax_approx.plot([], [], 'C3--', lw=2.0, label='NN Pred')
        self._scatter_tr  = self._ax_approx.scatter([], [], c='k', s=15, alpha=0.3,
                                                     label='Train data', zorder=2)
        self._ax_approx.legend(fontsize=9)
        self._line_loss,  = self._ax_loss.plot([], [], 'C2-', lw=1.8)
        self._err_fill    = None
        self._line_err,   = self._ax_err.plot([], [], 'C3-', lw=1.5)
        self._ax_hist     = None
        self.mpl.draw()

    def _init_4panel(self):
        """FR-MOD01-15: 4-panel layout for F-04 (adds error histogram)."""
        gs = self.mpl.fresh_gridspec(1, 4, wspace=0.38)
        fig = self.mpl.figure
        self._ax_approx = fig.add_subplot(gs[0, 0])
        self._ax_loss   = fig.add_subplot(gs[0, 1])
        self._ax_err    = fig.add_subplot(gs[0, 2])
        self._ax_hist   = fig.add_subplot(gs[0, 3])

        for ax, title, xl, yl in [
            (self._ax_approx, 'Function Approximation', 'x', 'f(x)'),
            (self._ax_loss,   'Training Loss',           'Epoch', 'MSE'),
            (self._ax_err,    'Absolute Error',          'x', '|error|'),
            (self._ax_hist,   'Error Distribution',      '|error|', 'Count'),
        ]:
            ax.set_title(title, fontweight='bold', fontsize=9)
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, alpha=0.25)
        self._ax_loss.set_yscale('log')
        self._ax_loss.grid(True, alpha=0.25, which='both')

        self._line_true,  = self._ax_approx.plot([], [], 'C0-',  lw=2.5, alpha=0.7, label='True')
        self._line_pred,  = self._ax_approx.plot([], [], 'C3--', lw=2.0, label='NN Pred')
        self._scatter_tr  = self._ax_approx.scatter([], [], c='k', s=15, alpha=0.3,
                                                     label='Train data', zorder=2)
        self._ax_approx.legend(fontsize=9)
        self._line_loss,  = self._ax_loss.plot([], [], 'C2-', lw=1.8)
        self._err_fill    = None
        self._line_err,   = self._ax_err.plot([], [], 'C3-', lw=1.5)
        self.mpl.draw()

    # ── BaseModule interface ──────────────────────────────────────────────────
    def _build_model(self):
        arch = ModelFactory.ARCH_OPTIONS[self._arch_cb.currentText()]
        act  = self._act_cb.currentText()
        lr   = self._lr_sl.value
        return ModelFactory.function_approximator(arch, act, lr)

    def _generate_data(self):
        func_name = self._func_cb.currentText()
        X_tr, y_tr, X_te, y_te = DataGenerators.function_approximation(func_name)
        self._X_te = X_te; self._y_te = y_te; self._X_tr = X_tr
        self._current_func = func_name

        # Switch panel layout when entering/leaving F-04
        is_f04 = (func_name == 'extreme')
        if is_f04 != self._is_f04:
            self._is_f04 = is_f04
            if is_f04: self._init_4panel()
            else:      self._init_3panel()
            self._epochs_buf.clear(); self._losses_buf.clear()
        return X_tr, y_tr      # BaseModule.run() uses data[0], data[1]

    def _get_training_config(self) -> TrainingConfig:
        epochs = int(self._ep_sl.value)
        func   = self._func_cb.currentText()
        if func == 'extreme':
            # FR-MOD01-14: F-04 special callbacks
            return TrainingConfig(
                epochs=epochs,
                use_reduce_lr=True,
                reduce_lr_factor=0.8,
                reduce_lr_patience=100,
                reduce_lr_min_lr=1e-5,
                use_early_stopping=True,
                early_stopping_patience=500,
            )
        # FR-MOD01-12: standard callbacks
        return TrainingConfig(
            epochs=epochs,
            use_reduce_lr=True,
            reduce_lr_factor=0.9,
            reduce_lr_patience=100,
            reduce_lr_min_lr=1e-5,
        )

    # ── live updates (FR-MOD01-09) ────────────────────────────────────────────
    def _on_progress(self, epoch: int, loss: float, val_loss: float):
        super()._on_progress(epoch, loss, val_loss)
        self._epochs_buf.append(epoch); self._losses_buf.append(loss)
        self._line_loss.set_data(self._epochs_buf, self._losses_buf)
        self._ax_loss.relim(); self._ax_loss.autoscale_view()
        self.mpl.draw_idle()

    # ── post-training render ──────────────────────────────────────────────────
    def _on_training_finished_impl(self, model, history):
        X_te    = self._X_te; y_te = self._y_te
        y_pred  = model.predict(X_te, verbose=0)
        mse     = float(np.mean((y_pred - y_te) ** 2))
        mae     = float(np.mean(np.abs(y_pred - y_te)))
        max_err = float(np.max(np.abs(y_pred - y_te)))
        x_flat  = X_te.flatten(); y_te_fl = y_te.flatten(); y_pred_fl = y_pred.flatten()
        error   = np.abs(y_pred_fl - y_te_fl)

        # Approximation plot — FR-MOD01-10 & FR-MOD01-11
        self._line_true.set_data(x_flat, y_te_fl)
        self._line_pred.set_data(x_flat, y_pred_fl)
        # Training scatter with stride=10 (FR-MOD01-10)
        f        = DataGenerators.FUNC_MAP[self._current_func]
        x_sc     = self._X_tr[::10].flatten()
        y_sc     = f(self._X_tr[::10]).flatten()
        self._scatter_tr.set_offsets(np.column_stack([x_sc, y_sc]))
        func_label = DataGenerators.FUNC_LABELS[self._current_func]
        self._ax_approx.set_title(
            f'{func_label}\nMSE: {mse:.6f}  MAE: {mae:.6f}  MaxErr: {max_err:.6f}',  # FR-MOD01-11
            fontweight='bold', fontsize=9)
        self._ax_approx.relim(); self._ax_approx.autoscale_view()

        # Full loss history
        full_loss = history.history['loss']
        self._line_loss.set_data(list(range(1, len(full_loss)+1)), full_loss)
        self._ax_loss.set_title(f'Loss — final: {full_loss[-1]:.2e}', fontweight='bold', fontsize=9)
        self._ax_loss.relim(); self._ax_loss.autoscale_view()

        # Error fill
        if self._err_fill: self._err_fill.remove()
        self._err_fill = self._ax_err.fill_between(x_flat, 0, error, color='C3', alpha=0.25)
        self._line_err.set_data(x_flat, error)
        self._ax_err.set_title(f'Max Error: {max_err:.2e}', fontweight='bold', fontsize=9)
        self._ax_err.relim(); self._ax_err.autoscale_view()

        # FR-MOD01-15: F-04 error histogram
        if self._is_f04 and self._ax_hist is not None:
            self._ax_hist.cla()
            self._ax_hist.hist(error, bins=40, color='C3', alpha=0.7, edgecolor='none')
            self._ax_hist.axvline(mae, color='C0', lw=2, ls='--', label=f'MAE={mae:.4f}')
            self._ax_hist.set_title('Error Distribution', fontweight='bold', fontsize=9)
            self._ax_hist.set_xlabel('|error|'); self._ax_hist.set_ylabel('Count')
            self._ax_hist.legend(fontsize=8); self._ax_hist.grid(True, alpha=0.25)

        self._mse = mse; self._mae = mae
        self.mpl.draw()

    def _on_reset_impl(self):
        self._epochs_buf.clear(); self._losses_buf.clear()
        self._is_f04 = False; self._init_3panel()

    def get_param_values(self):
        return {**self._pg_func.values(), **self._pg_arch.values(), **self._pg_train.values()}

    def get_metrics(self):
        return {'mse': getattr(self,'_mse',None), 'mae': getattr(self,'_mae',None)}
