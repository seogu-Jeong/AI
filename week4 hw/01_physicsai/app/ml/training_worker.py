"""
TrainingWorker — QThread-based training executor.
Communicates exclusively via Qt signals; never touches QWidget instances.
"""
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QThread, Signal
import numpy as np


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 32
    validation_split: float = 0.2
    log_interval: int = 10
    use_reduce_lr: bool = True
    reduce_lr_factor: float = 0.8
    reduce_lr_patience: int = 50
    reduce_lr_min_lr: float = 1e-6
    use_early_stopping: bool = False
    early_stopping_patience: int = 300


class TrainingWorker(QThread):
    """
    Runs Keras model.fit() in a background thread.

    Thread-safety guarantees (TRD-01 §4.2):
      - Never accesses QWidget
      - Emits only primitive types / numpy arrays
      - Responds to request_stop() within one epoch
    """
    progress_updated  = Signal(int, float, float)   # epoch, loss, val_loss
    training_finished = Signal(object, object)       # model, history
    training_error    = Signal(str)

    def __init__(self, model, X: np.ndarray, y: np.ndarray,
                 config: TrainingConfig, parent=None):
        super().__init__(parent)
        self.model      = model
        self.X          = X.copy()
        self.y          = y.copy()
        self.config     = config
        self._stop_flag = False

    def request_stop(self):
        self._stop_flag = True

    def run(self):
        try:
            from tensorflow import keras
            callbacks = self._build_callbacks(keras)
            history = self.model.fit(
                self.X, self.y,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=0,
            )
            if not self._stop_flag:
                self.training_finished.emit(self.model, history)
        except Exception as exc:
            import traceback
            self.training_error.emit(traceback.format_exc())

    def _build_callbacks(self, keras):
        """
        Build callbacks list.
        _QtProgressCallback is constructed as a proper keras.callbacks.Callback
        subclass here, after keras is available in the worker thread.
        """
        worker   = self
        total    = self.config.epochs
        interval = self.config.log_interval

        # Dynamically subclass keras.callbacks.Callback so Keras accepts it.
        class _QtProgressCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch: int, logs=None):
                if worker._stop_flag:
                    self.model.stop_training = True
                    return
                if (epoch + 1) % interval == 0 or epoch == total - 1:
                    logs   = logs or {}
                    loss   = float(logs.get('loss', 0.0))
                    v_loss = float(logs.get('val_loss', loss))
                    worker.progress_updated.emit(epoch + 1, loss, v_loss)

        cbs = [_QtProgressCallback()]

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
