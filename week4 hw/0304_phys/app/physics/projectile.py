"""
ProjectilePhysics — vacuum and air-resistance (quadratic drag) simulation.
State vector: [x, y, vx, vy]. No Qt imports.
SDD-PHYSAI-003 §2.2
"""
import numpy as np
from .rk4 import rk4_integrate


class ProjectilePhysics:
    G = 9.81  # m/s²

    def __init__(self, k: float = 0.0):
        """k: drag coefficient kg⁻¹. k=0 → vacuum."""
        assert k >= 0
        self.k = k

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, vx, vy = state
        v_mag = np.sqrt(vx**2 + vy**2)
        drag  = self.k * v_mag
        return np.array([vx, vy, -drag * vx, -self.G - drag * vy])

    def simulate(self, v0: float, angle_deg: float, dt: float = 0.01) -> np.ndarray:
        """Simulates until y < 0. Returns (N, 4) array [x, y, vx, vy]."""
        theta  = np.deg2rad(angle_deg)
        state0 = np.array([0.0, 0.0, v0 * np.cos(theta), v0 * np.sin(theta)])
        traj   = rk4_integrate(self.derivatives, state0, (0.0, 1000.0), dt,
                                stop_condition=lambda s: s[1] < 0.0)
        return traj

    def landing_range(self, v0: float, angle_deg: float, dt: float = 0.01) -> float:
        """x-coordinate at landing using linear interpolation to find y=0 crossing."""
        traj = self.simulate(v0, angle_deg, dt)
        if len(traj) >= 2 and traj[-1, 1] < 0:
            x0, y0 = traj[-2, 0], traj[-2, 1]
            x1, y1 = traj[-1, 0], traj[-1, 1]
            if y0 > y1:
                frac = y0 / (y0 - y1)
                return float(x0 + frac * (x1 - x0))
        return float(traj[-1, 0])

    def vacuum_range(self, v0: float, angle_deg: float) -> float:
        """Analytical range: R = v₀²sin(2θ)/g. Only valid when k=0."""
        theta = np.deg2rad(angle_deg)
        return (v0**2 * np.sin(2 * theta)) / self.G
