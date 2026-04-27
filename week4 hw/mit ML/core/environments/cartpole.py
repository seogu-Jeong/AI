from __future__ import annotations
import numpy as np


class CartPole:
    """CartPole environment with Euler integration.

    State: [x, x_dot, theta, theta_dot]
    Actions: 0=push left, 1=push right
    """

    GRAVITY = 9.8
    MASSCART = 1.0
    MASSPOLE = 0.1
    LENGTH = 0.5       # half-pole length
    FORCE_MAG = 10.0
    TAU = 0.02         # seconds per step

    THETA_LIMIT = 0.2094   # ~12 degrees
    X_LIMIT = 2.4

    def __init__(self):
        self.n_actions = 2
        self._state = np.zeros(4)

    def reset(self) -> np.ndarray:
        self._state = np.random.uniform(-0.05, 0.05, size=4)
        return self._state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        x, x_dot, theta, theta_dot = self._state
        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG

        total_mass = self.MASSCART + self.MASSPOLE
        pole_mass_length = self.MASSPOLE * self.LENGTH
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        temp = (force + pole_mass_length * theta_dot ** 2 * sin_theta) / total_mass
        theta_acc = (self.GRAVITY * sin_theta - cos_theta * temp) / (
            self.LENGTH * (4.0 / 3.0 - self.MASSPOLE * cos_theta ** 2 / total_mass)
        )
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass

        # Euler integration
        x = x + self.TAU * x_dot
        x_dot = x_dot + self.TAU * x_acc
        theta = theta + self.TAU * theta_dot
        theta_dot = theta_dot + self.TAU * theta_acc

        self._state = np.array([x, x_dot, theta, theta_dot])

        done = (abs(x) > self.X_LIMIT or abs(theta) > self.THETA_LIMIT)
        reward = 0.0 if done else 1.0
        return self._state.copy(), reward, done

    def discretize(self, state: np.ndarray, bins: int = 10) -> int:
        """Map continuous state to integer index (bins^4 states total)."""
        x, x_dot, theta, theta_dot = state
        limits = [
            (-self.X_LIMIT, self.X_LIMIT),
            (-3.0, 3.0),
            (-self.THETA_LIMIT, self.THETA_LIMIT),
            (-3.0, 3.0),
        ]
        idx = 0
        for i, (val, (lo, hi)) in enumerate(zip([x, x_dot, theta, theta_dot], limits)):
            bucket = int((np.clip(val, lo, hi) - lo) / (hi - lo) * bins)
            bucket = min(bucket, bins - 1)
            idx = idx * bins + bucket
        return idx
