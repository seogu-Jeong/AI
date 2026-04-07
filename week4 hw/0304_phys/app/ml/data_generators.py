"""
DataGenerators — synthetic training data for MOD-02, MOD-04, MOD-05.
Pure NumPy + physics engine. No Qt imports.
SDD-PHYSAI-003 §3.3
"""
import numpy as np
from typing import Optional, Callable


class DataGenerators:

    @staticmethod
    def projectile(n_samples: int = 2000, noise_m: float = 0.5):
        """
        MOD-02: FR-MOD02-01..03
        v₀∈U[10,50], θ∈U[20°,70°], t∈U[0, 0.9·T_f]; y<0 filtered.
        Returns X(n,3)=[v0,theta,t], Y(n,2)=[x,y].
        """
        G   = 9.81
        rng = np.random.default_rng()
        v0    = rng.uniform(10, 50, n_samples)
        theta = rng.uniform(20, 70, n_samples)
        t_max = 2 * v0 * np.sin(np.deg2rad(theta)) / G
        t     = rng.uniform(0, 0.9 * t_max, n_samples)
        x = v0 * np.cos(np.deg2rad(theta)) * t + rng.normal(0, noise_m, n_samples)
        y = (v0 * np.sin(np.deg2rad(theta)) * t
             - 0.5 * G * t**2
             + rng.normal(0, noise_m, n_samples))
        mask = y >= 0
        X = np.column_stack([v0[mask], theta[mask], t[mask]])
        Y = np.column_stack([x[mask], y[mask]])
        return X, Y

    @staticmethod
    def projectile_test(n_samples: int = 500):
        """FR-MOD02-04: noise-free test set."""
        G   = 9.81
        rng = np.random.default_rng(seed=0)
        v0    = rng.uniform(10, 50, n_samples)
        theta = rng.uniform(20, 70, n_samples)
        t_max = 2 * v0 * np.sin(np.deg2rad(theta)) / G
        t     = rng.uniform(0, 0.9 * t_max, n_samples)
        x = v0 * np.cos(np.deg2rad(theta)) * t
        y = v0 * np.sin(np.deg2rad(theta)) * t - 0.5 * G * t**2
        mask = y >= 0
        X = np.column_stack([v0[mask], theta[mask], t[mask]])
        Y = np.column_stack([x[mask], y[mask]])
        return X, Y

    @staticmethod
    def pendulum(n_samples: int = 2000, noise_frac: float = 0.01):
        """
        MOD-04: FR-MOD04-03
        L∈U[0.5,3.0]m, θ₀∈U[5°,80°]; multiplicative noise N(1, noise_frac).
        Returns X(n,2)=[L,theta0_deg], y(n,1)=[T_seconds].
        """
        from app.physics.pendulum import PendulumPhysics
        rng    = np.random.default_rng()
        L      = rng.uniform(0.5, 3.0, n_samples)
        theta0 = rng.uniform(5.0, 80.0, n_samples)
        T_true = np.array([PendulumPhysics.true_period(l, t) for l, t in zip(L, theta0)])
        T_noisy = T_true * (1 + rng.normal(0, noise_frac, n_samples))
        X = np.column_stack([L, theta0])
        y = T_noisy.reshape(-1, 1)
        return X, y

    @staticmethod
    def air_resistance(n_samples: int = 2000, k: float = 0.05,
                       progress_callback: Optional[Callable[[int], None]] = None):
        """
        MOD-05: FR-MOD05-07
        v₀∈U[10,100], θ∈U[10°,80°]; runs RK4 for each sample.
        progress_callback(i) called every 100 samples for LoadingWorker progress bar.
        Returns X(n,2)=[v0,angle], y(n,1)=[range_m].
        """
        from app.physics.projectile import ProjectilePhysics
        rng    = np.random.default_rng()
        phys   = ProjectilePhysics(k=k)
        v0s    = rng.uniform(10, 100, n_samples)
        angles = rng.uniform(10, 80,  n_samples)
        ranges = np.zeros(n_samples)
        for i, (v, a) in enumerate(zip(v0s, angles)):
            ranges[i] = phys.landing_range(v, a)
            if progress_callback is not None and i % 100 == 0:
                progress_callback(i)
        X = np.column_stack([v0s, angles])
        y = ranges.reshape(-1, 1)
        return X, y
