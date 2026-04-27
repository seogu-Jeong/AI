"""
DataGenerators — synthetic training data for each module.
Pure NumPy, no Qt imports.
"""
import numpy as np
from typing import Callable


class DataGenerators:

    FUNC_MAP = {
        'sin(x)':            lambda x: np.sin(x),
        'cos(x)+0.5sin(2x)': lambda x: np.cos(x) + 0.5 * np.sin(2 * x),
        'x·sin(x)':          lambda x: x * np.sin(x),
        'extreme':           lambda x: (np.sin(x) + 0.5 * np.sin(2 * x)
                                        + 0.3 * np.cos(3 * x) + 0.2 * np.sin(5 * x)
                                        + 0.1 * x * np.cos(x)),
    }

    @staticmethod
    def function_approximation(func_name: str, n_train: int = 300, n_test: int = 500):
        f      = DataGenerators.FUNC_MAP[func_name]
        domain = (-3 * np.pi, 3 * np.pi) if func_name == 'extreme' else (-2 * np.pi, 2 * np.pi)
        X_tr   = np.linspace(*domain, n_train).reshape(-1, 1)
        X_te   = np.linspace(*domain, n_test).reshape(-1, 1)
        return X_tr, f(X_tr), X_te, f(X_te)

    @staticmethod
    def overfitting(n_train: int = 100, n_val: int = 50, n_test: int = 200,
                    noise: float = 0.3):
        f   = lambda x: np.sin(2 * x) + 0.5 * x
        rng = np.random.default_rng(seed=42)
        X_tr  = rng.uniform(-2, 2, n_train).reshape(-1, 1)
        X_val = rng.uniform(-2, 2, n_val).reshape(-1, 1)
        X_te  = np.linspace(-2, 2, n_test).reshape(-1, 1)
        return (X_tr,  f(X_tr)  + rng.normal(0, noise, (n_train, 1)),
                X_val, f(X_val) + rng.normal(0, noise, (n_val, 1)),
                X_te,  f(X_te))

    @staticmethod
    def projectile(n_samples: int = 2000, noise_m: float = 0.5):
        G   = 9.81
        rng = np.random.default_rng()
        v0    = rng.uniform(10, 50, n_samples)
        theta = rng.uniform(20, 70, n_samples)
        t_max = 2 * v0 * np.sin(np.deg2rad(theta)) / G
        t     = rng.uniform(0, 0.9 * t_max, n_samples)
        x = v0 * np.cos(np.deg2rad(theta)) * t + rng.normal(0, noise_m, n_samples)
        y = v0 * np.sin(np.deg2rad(theta)) * t - 0.5 * G * t ** 2 + rng.normal(0, noise_m, n_samples)
        mask = y >= 0
        X = np.column_stack([v0[mask], theta[mask], t[mask]])
        Y = np.column_stack([x[mask], y[mask]])
        return X, Y

    @staticmethod
    def pendulum(n_samples: int = 2000, noise_frac: float = 0.01):
        from app.physics.pendulum import PendulumPhysics
        rng    = np.random.default_rng()
        L      = rng.uniform(0.5, 3.0, n_samples)
        theta0 = rng.uniform(5.0, 80.0, n_samples)
        T_true = np.array([PendulumPhysics.true_period(l, t) for l, t in zip(L, theta0)])
        T_noisy = T_true * (1 + rng.normal(0, noise_frac, n_samples))
        return np.column_stack([L, theta0]), T_noisy.reshape(-1, 1)

    @staticmethod
    def air_resistance(n_samples: int = 1000, k: float = 0.05,
                       progress_cb=None):
        from app.physics.projectile import ProjectilePhysics
        rng    = np.random.default_rng()
        physics = ProjectilePhysics(k=k)
        v0s    = rng.uniform(10, 80, n_samples)
        angles = rng.uniform(15, 75, n_samples)
        ranges = np.zeros(n_samples)
        for i, (v, a) in enumerate(zip(v0s, angles)):
            ranges[i] = physics.landing_range(v, a)
            if progress_cb and i % 50 == 0:
                progress_cb(i)
        return np.column_stack([v0s, angles]), ranges.reshape(-1, 1)
