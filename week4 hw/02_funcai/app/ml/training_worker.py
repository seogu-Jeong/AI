"""
TrainingWorker — QThread training executor.
Signal-only communication; never touches QWidget instances.
"""
from dataclasses import dataclass
from PySide6.QtCore import QThread, Signal
import numpy as np


@dataclass
class TrainingConfig:
    epochs: int = 3000
    batch_size: int = 32
    validation_split: float = 0.0   # MOD-03 passes its own val data
    log_interval: int = 10
    use_reduce_lr: bool = True
    reduce_lr_factor: float = 0.9   # FR-MOD01-12
    reduce_lr_patience: int = 100   # FR-MOD01-12
    reduce_lr_min_lr: float = 1e-5  # FR-MOD01-12
    use_early_stopping: bool = False
    early_stopping_patience: int = 500  # FR-MOD01-14


class TrainingWorker(QThread):
    progress_updated  = Signal(int, float, float)  # epoch, loss, val_loss
    training_finished = Signal(object, object)      # model, history
    training_error    = Signal(str)

    def __init__(self, model, X: np.ndarray, y: np.ndarray,
                 config: TrainingConfig,
                 X_val: np.ndarray = None, y_val: np.ndarray = None,
                 parent=None):
        super().__init__(parent)
        self.model      = model
        self.X          = X.copy()
        self.y          = y.copy()
        self.X_val      = X_val.copy() if X_val is not None else None
        self.y_val      = y_val.copy() if y_val is not None else None
        self.config     = config
        self._stop_flag = False

    def request_stop(self):
        self._stop_flag = True

    def run(self):
        try:
            from tensorflow import keras
            callbacks = self._build_callbacks(keras)
            fit_kwargs = dict(
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=0,
            )
            if self.X_val is not None:
                fit_kwargs['validation_data'] = (self.X_val, self.y_val)
            elif self.config.validation_split > 0:
                fit_kwargs['validation_split'] = self.config.validation_split
            history = self.model.fit(self.X, self.y, **fit_kwargs)
            if not self._stop_flag:
                self.training_finished.emit(self.model, history)
        except Exception:
            import traceback
            self.training_error.emit(traceback.format_exc())

    def _build_callbacks(self, keras):
        worker   = self
        total    = self.config.epochs
        interval = self.config.log_interval

        class _Cb(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if worker._stop_flag:
                    self.model.stop_training = True
                    return
                if (epoch + 1) % interval == 0 or epoch == total - 1:
                    logs   = logs or {}
                    loss   = float(logs.get('loss', 0.0))
                    v_loss = float(logs.get('val_loss', loss))
                    worker.progress_updated.emit(epoch + 1, loss, v_loss)

        cbs = [_Cb()]
        if self.config.use_reduce_lr:
            cbs.append(keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.reduce_lr_min_lr,
                verbose=0,
            ))
        if self.config.use_early_stopping:
            cbs.append(keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=0,
            ))
        return cbs
