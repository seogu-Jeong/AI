"""
DataGenerators — synthetic training data for MOD-01 and MOD-03.
Pure NumPy, no Qt imports.
"""
import numpy as np


class DataGenerators:

    FUNC_MAP = {
        'sin(x)':            lambda x: np.sin(x),
        'cos(x)+0.5sin(2x)': lambda x: np.cos(x) + 0.5 * np.sin(2 * x),
        'x·sin(x)':          lambda x: x * np.sin(x),
        'extreme':           lambda x: (np.sin(x) + 0.5 * np.sin(2 * x)
                                        + 0.3 * np.cos(3 * x) + 0.2 * np.sin(5 * x)
                                        + 0.1 * x * np.cos(x)),
    }
    FUNC_LABELS = {
        'sin(x)':            'F-01: sin(x)',
        'cos(x)+0.5sin(2x)': 'F-02: cos(x) + 0.5·sin(2x)',
        'x·sin(x)':          'F-03: x·sin(x)',
        'extreme':           'F-04: sin(x)+0.5sin(2x)+0.3cos(3x)+0.2sin(5x)+0.1x·cos(x)',
    }

    @staticmethod
    def function_approximation(func_name: str, n_test: int = 400):
        """
        FR-MOD01-08: 200 train pts; FR-MOD01-13: F-04 uses 500 pts over [-3π,3π].
        Returns X_train, y_train, X_test, y_test — all shape (n,1).
        """
        f = DataGenerators.FUNC_MAP[func_name]
        if func_name == 'extreme':
            domain  = (-3 * np.pi, 3 * np.pi)
            n_train = 500   # FR-MOD01-13
        else:
            domain  = (-2 * np.pi, 2 * np.pi)
            n_train = 200   # FR-MOD01-08

        X_tr = np.linspace(*domain, n_train).reshape(-1, 1)
        X_te = np.linspace(*domain, n_test).reshape(-1, 1)
        return X_tr, f(X_tr), X_te, f(X_te)

    @staticmethod
    def overfitting(n_train: int = 100, n_val: int = 50,
                    n_test: int = 200, noise: float = 0.3):
        """
        FR-MOD03-01: identical train/val/test splits; test is noise-free.
        Returns X_tr, y_tr, X_val, y_val, X_te, y_te.
        """
        f   = lambda x: np.sin(2 * x) + 0.5 * x
        rng = np.random.default_rng(seed=42)
        X_tr  = rng.uniform(-2, 2, n_train).reshape(-1, 1)
        X_val = rng.uniform(-2, 2, n_val).reshape(-1, 1)
        X_te  = np.linspace(-2, 2, n_test).reshape(-1, 1)   # FR-MOD03-01: noise-free
        return (X_tr,  f(X_tr)  + rng.normal(0, noise, (n_train, 1)),
                X_val, f(X_val) + rng.normal(0, noise, (n_val, 1)),
                X_te,  f(X_te))
