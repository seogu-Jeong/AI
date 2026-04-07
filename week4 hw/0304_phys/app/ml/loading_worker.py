"""
LoadingWorker — QThread for slow data generation (MOD-05: 2000 RK4 sims).
FR-MOD05-08: progress bar updates every 100 simulations.
"""
from PySide6.QtCore import QThread, Signal
import numpy as np


class LoadingWorker(QThread):
    progress_updated = Signal(int)    # samples completed (0..n_samples)
    data_ready       = Signal(object, object)  # X, y numpy arrays
    loading_error    = Signal(str)

    def __init__(self, n_samples: int = 2000, k: float = 0.05, parent=None):
        super().__init__(parent)
        self.n_samples   = n_samples
        self.k           = k
        self._stop_flag  = False

    def request_stop(self):
        self._stop_flag = True

    def run(self):
        try:
            from app.ml.data_generators import DataGenerators

            def _cb(i):
                if self._stop_flag:
                    raise InterruptedError("Stopped by user")
                self.progress_updated.emit(i)

            X, y = DataGenerators.air_resistance(
                n_samples=self.n_samples, k=self.k, progress_callback=_cb)
            if not self._stop_flag:
                self.progress_updated.emit(self.n_samples)
                self.data_ready.emit(X, y)
        except InterruptedError:
            pass
        except Exception:
            import traceback
            self.loading_error.emit(traceback.format_exc())
