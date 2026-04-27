"""
MOD-01 — 1D Function Approximation
Demonstrates Universal Approximation Theorem via neural network regression.
"""
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout

from app.ml.data_generators import DataGenerators
from app.ml.models import ModelFactory
from app.ml.training_worker import TrainingConfig
from app.widgets.matplotlib_widget import MatplotlibWidget
from app.widgets.param_group import ParamGroup
from .base_module import BaseModule

ARCH_MAP = {
    'Shallow [128]':          [128],
    'Medium [128×128]':       [128, 128],
    'Deep [256×256×128×64]':  [256, 256, 128, 64],
    'Tiny [32×16]':           [32, 16],
}

FUNC_OPTIONS = list(DataGenerators.FUNC_MAP.keys())


class FunctionApproximationModule(BaseModule):
    MODULE_ID   = "MOD-01"
    MODULE_NAME = "1D Function Approximation"
    MODULE_DESC = "Neural network learns to approximate mathematical functions (Universal Approximation Theorem)"

    def _setup_param_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Function selection
        self._pg_func = ParamGroup("Target Function")
        self._func_cb = self._pg_func.add_combo(
            'func', "Function:", FUNC_OPTIONS,
            tooltip="Mathematical function to approximate")
        layout.addWidget(self._pg_func)

        # Architecture
        self._pg_arch = ParamGroup("Network Architecture")
        arch_opts = list(ARCH_MAP.keys())
        self._arch_cb = self._pg_arch.add_combo(
            'arch', "Architecture:", arch_opts, default_idx=1,
            tooltip="Hidden layer configuration — wider/deeper ≈ more capacity")
        self._act_cb = self._pg_arch.add_combo(
            'activation', "Activation:", ['tanh', 'relu', 'sigmoid'], default_idx=0,
            tooltip="tanh: best for periodic functions; relu: faster but may oscillate")
        layout.addWidget(self._pg_arch)

        # Training
        self._pg_train = ParamGroup("Training")
        self._ep_slider = self._pg_train.add_slider(
            'epochs', label="Epochs", min_val=100, max_val=5000, default=1000,
            step=100, unit="", decimals=0,
            tooltip="Number of gradient descent steps")
        self._lr_slider = self._pg_train.add_slider(
            'lr', label="Learning Rate", min_val=0.001, max_val=0.1, default=0.01,
            step=0.001, unit="", decimals=3,
            tooltip="Adam optimiser initial learning rate")
        layout.addWidget(self._pg_train)

        # Wire dirty state
        for pg in (self._pg_func, self._pg_arch, self._pg_train):
            pg.any_value_changed.connect(lambda *_: self._mark_dirty())
        return panel

    def _setup_plot_area(self) -> QWidget:
        self.mpl = MatplotlibWidget(figsize=(15, 5))
        self._epochs_buf, self._losses_buf = [], []
        axes = self.mpl.fresh_axes(nrows=1, ncols=3)
        self._ax_approx, self._ax_loss, self._ax_err = axes
        self._init_artists()
        return self.mpl

    def _init_artists(self):
        dark = False  # will sync via ThemeManager later
        for ax, title in [(self._ax_approx, 'Function Approximation'),
                           (self._ax_loss,   'Training Loss'),
                           (self._ax_err,    'Absolute Error')]:
            ax.set_title(title, fontweight='bold', pad=8)
            ax.grid(True, alpha=0.25)

        self._ax_approx.set_xlabel('x'); self._ax_approx.set_ylabel('f(x)')
        self._ax_loss.set_xlabel('Epoch'); self._ax_loss.set_ylabel('MSE (log)')
        self._ax_loss.set_yscale('log')
        self._ax_loss.grid(True, alpha=0.25, which='both')
        self._ax_err.set_xlabel('x'); self._ax_err.set_ylabel('|error|')

        self._line_true,  = self._ax_approx.plot([], [], 'C0-',  lw=2.5, label='True',  alpha=0.8)
        self._line_pred,  = self._ax_approx.plot([], [], 'C3--', lw=2.0, label='NN Pred')
        self._line_train  = self._ax_approx.scatter([], [], c='#555', s=12, alpha=0.3, label='Train pts', zorder=2)
        self._ax_approx.legend(fontsize=9)

        self._line_loss,  = self._ax_loss.plot([], [], 'C2-', lw=1.8)
        self._err_fill    = None
        self._line_err,   = self._ax_err.plot([], [], 'C3-', lw=1.5)

        self.mpl.draw()

    # ── BaseModule interface ─────────────────────────────────────────────────
    def _build_model(self):
        arch = ARCH_MAP[self._arch_cb.currentText()]
        act  = self._act_cb.currentText()
        lr   = self._lr_slider.value
        return ModelFactory.function_approximator(arch, act, lr)

    def _generate_data(self):
        func_name = self._func_cb.currentText()
        X_tr, y_tr, X_te, y_te = DataGenerators.function_approximation(func_name)
        self._X_te = X_te
        self._y_te = y_te
        self._X_tr = X_tr
        return X_tr, y_tr

    def _get_training_config(self) -> TrainingConfig:
        return TrainingConfig(epochs=int(self._ep_slider.value))

    def _on_progress(self, epoch: int, loss: float, val_loss: float):
        super()._on_progress(epoch, loss, val_loss)
        cfg = self._get_training_config()
        self._epochs_buf.append(epoch)
        self._losses_buf.append(loss)
        self._line_loss.set_data(self._epochs_buf, self._losses_buf)
        self._ax_loss.relim(); self._ax_loss.autoscale_view()
        self.mpl.draw_idle()

    def _on_training_finished_impl(self, model, history):
        func_name = self._func_cb.currentText()
        y_pred  = model.predict(self._X_te, verbose=0)
        mse     = float(np.mean((y_pred - self._y_te) ** 2))
        mae     = float(np.mean(np.abs(y_pred - self._y_te)))
        max_err = float(np.max(np.abs(y_pred - self._y_te)))

        x_flat   = self._X_te.flatten()
        y_te_fl  = self._y_te.flatten()
        y_pred_fl = y_pred.flatten()
        error    = np.abs(y_pred_fl - y_te_fl)

        # Approximation plot
        self._line_true.set_data(x_flat, y_te_fl)
        self._line_pred.set_data(x_flat, y_pred_fl)
        # Update scatter train data
        self._line_train.set_offsets(
            np.column_stack([self._X_tr.flatten(),
                             DataGenerators.FUNC_MAP[func_name](self._X_tr).flatten()]))
        self._ax_approx.set_title(
            f'{func_name}\nMSE={mse:.2e}  MAE={mae:.2e}  MaxErr={max_err:.2e}',
            fontweight='bold', fontsize=9)
        self._ax_approx.relim(); self._ax_approx.autoscale_view()

        # Loss (final)
        full_loss = history.history['loss']
        epochs    = list(range(1, len(full_loss) + 1))
        self._line_loss.set_data(epochs, full_loss)
        self._ax_loss.relim(); self._ax_loss.autoscale_view()
        self._ax_loss.set_title(f'Loss (final: {full_loss[-1]:.2e})', fontweight='bold')

        # Error fill
        if self._err_fill:
            self._err_fill.remove()
        self._err_fill = self._ax_err.fill_between(x_flat, 0, error, color='C3', alpha=0.25)
        self._line_err.set_data(x_flat, error)
        self._ax_err.set_title(f'Max Error: {max_err:.2e}', fontweight='bold')
        self._ax_err.relim(); self._ax_err.autoscale_view()

        # Store metrics
        self._mse, self._mae = mse, mae
        self.mpl.draw()

    def _on_reset_impl(self):
        self._epochs_buf.clear(); self._losses_buf.clear()
        for ax in (self._ax_approx, self._ax_loss, self._ax_err):
            ax.cla()
        self._init_artists()

    def get_param_values(self):
        return {**self._pg_func.values(), **self._pg_arch.values(), **self._pg_train.values()}

    def get_metrics(self):
        return {'mse': getattr(self, '_mse', None), 'mae': getattr(self, '_mae', None)}
