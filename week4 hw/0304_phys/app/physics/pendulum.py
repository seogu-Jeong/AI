"""
PendulumPhysics — simple pendulum RK4 simulation and period formulas.
State vector: [theta (rad), omega (rad/s)]. No Qt imports.
SDD-PHYSAI-003 §2.3
"""
import numpy as np
from .rk4 import rk4_integrate


class PendulumPhysics:
    G = 9.81  # m/s²

    def __init__(self, L: float, g: float = 9.81):
        assert L > 0
        self.L = L
        self.g = g

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        theta, omega = state
        return np.array([omega, -(self.g / self.L) * np.sin(theta)])

    def simulate(self, theta0_deg: float, n_periods: float = 3.0,
                 dt: float = 0.01) -> np.ndarray:
        """Simulates n_periods of pendulum. Returns (N, 2): [theta_rad, omega_rad_s]."""
        theta0   = np.deg2rad(theta0_deg)
        T_approx = self.small_angle_period()
        t_max    = T_approx * (n_periods + 1)   # slight overrun for coverage
        state0   = np.array([theta0, 0.0])
        return rk4_integrate(self.derivatives, state0, (0.0, t_max), dt)

    def small_angle_period(self) -> float:
        """T_small = 2π√(L/g)"""
        return 2 * np.pi * np.sqrt(self.L / self.g)

    @staticmethod
    def true_period(L: float, theta0_deg: float, g: float = 9.81) -> float:
        """4th-order elliptic integral approximation. Error < 0.01% for θ₀ < 80°."""
        theta0     = np.deg2rad(theta0_deg)
        T_small    = 2 * np.pi * np.sqrt(L / g)
        correction = 1.0 + (1.0/16.0) * theta0**2 + (11.0/3072.0) * theta0**4
        return T_small * correction

    def energy(self, state: np.ndarray) -> float:
        """Total mechanical energy per unit mass: KE + PE."""
        theta, omega = state
        KE = 0.5 * self.L**2 * omega**2
        PE = self.g * self.L * (1 - np.cos(theta))
        return KE + PE
