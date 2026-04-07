"""
RK4 numerical integrator — pure NumPy, no Qt imports.
Global truncation error O(h^4).
"""
import numpy as np
from typing import Callable, Optional

ArrayF = np.ndarray  # shape (n,), dtype float64


def rk4_step(
    f: Callable[[ArrayF, float], ArrayF],
    state: ArrayF,
    t: float,
    dt: float,
) -> ArrayF:
    k1 = f(state,                  t)
    k2 = f(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(state + dt       * k3, t + dt)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_integrate(
    f: Callable[[ArrayF, float], ArrayF],
    state0: ArrayF,
    t_span: tuple,
    dt: float,
    stop_condition: Optional[Callable[[ArrayF], bool]] = None,
    max_steps: int = 100_000,
) -> np.ndarray:
    states = [state0.copy()]
    state  = state0.copy()
    t      = t_span[0]
    t_end  = t_span[1]
    step   = 0
    while t < t_end and step < max_steps:
        state = rk4_step(f, state, t, dt)
        t    += dt
        step += 1
        states.append(state.copy())
        if stop_condition is not None and stop_condition(state):
            break
    return np.array(states)
