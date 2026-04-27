"""
Projectile physics — pure NumPy, no Qt imports.
State: [x, y, vx, vy]
"""
import numpy as np
from .rk4 import rk4_integrate


class ProjectilePhysics:
    """2D projectile with optional quadratic air resistance.

    Equations of motion:
        dx/dt  = vx
        dy/dt  = vy
        dvx/dt = -k * |v| * vx
        dvy/dt = -g - k * |v| * vy
    """
    G = 9.81  # m/s²

    def __init__(self, k: float = 0.0):
        assert k >= 0, "Drag coefficient must be non-negative"
        self.k = k

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, vx, vy = state
        v_mag = np.sqrt(vx ** 2 + vy ** 2)
        drag  = self.k * v_mag
        return np.array([vx, vy, -drag * vx, -self.G - drag * vy])

    def simulate(self, v0: float, angle_deg: float, dt: float = 0.01) -> np.ndarray:
        theta  = np.deg2rad(angle_deg)
        state0 = np.array([0.0, 0.0, v0 * np.cos(theta), v0 * np.sin(theta)])
        return rk4_integrate(
            f=self.derivatives,
            state0=state0,
            t_span=(0.0, 1000.0),
            dt=dt,
            stop_condition=lambda s: s[1] < 0.0,
        )

    def landing_range(self, v0: float, angle_deg: float, dt: float = 0.01) -> float:
        traj = self.simulate(v0, angle_deg, dt)
        # Linear interpolation to y=0 between last two points
        if len(traj) >= 2 and traj[-1, 1] < 0:
            x0, y0 = traj[-2, 0], traj[-2, 1]
            x1, y1 = traj[-1, 0], traj[-1, 1]
            if y0 > y1:
                frac = y0 / (y0 - y1)
                return float(x0 + frac * (x1 - x0))
        return float(traj[-1, 0])

    def vacuum_range(self, v0: float, angle_deg: float) -> float:
        """Analytical range: R = v0^2 * sin(2θ) / g"""
        theta = np.deg2rad(angle_deg)
        return (v0 ** 2 * np.sin(2 * theta)) / self.G
