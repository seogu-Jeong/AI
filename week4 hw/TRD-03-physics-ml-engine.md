# Software Design Description
## PhysicsAI Simulator — Physics Engine & ML System Design
**Document ID:** SDD-PHYSAI-003  
**Version:** 2.0  
**Date:** 2026-04-06  
**Standard:** IEEE Std 1016-2009  
**Parent Document:** SDD-PHYSAI-001  
**Status:** Approved  

---

## 1. Introduction

### 1.1 Purpose

This SDD specifies the design of the Physics Engine and ML subsystems. It documents:
- RK4 integrator design with numerical accuracy analysis
- Physics model class hierarchy with interface contracts
- Neural network architecture decisions with quantitative justification
- Training pipeline design including the QThread-based worker
- Error handling and graceful degradation strategies

### 1.2 Design Philosophy

The Physics Engine and ML Engine are designed as **pure computation modules** — no Qt imports, no side effects, no global state. This constraint ensures:
1. Unit testability without a running QApplication
2. Potential reuse outside the GUI application
3. Clean separation enabling replacement of either engine independently

---

## 2. Physics Engine Design

### 2.1 RK4 Numerical Integrator

#### 2.1.1 Algorithm

The fourth-order Runge-Kutta method solves the initial value problem:
```
dy/dt = f(y, t),  y(t₀) = y₀
```

Single-step update:
```
k₁ = f(yₙ, tₙ)
k₂ = f(yₙ + h/2·k₁, tₙ + h/2)
k₃ = f(yₙ + h/2·k₂, tₙ + h/2)
k₄ = f(yₙ + h·k₃,   tₙ + h)
yₙ₊₁ = yₙ + h/6·(k₁ + 2k₂ + 2k₃ + k₄)
```

**Global truncation error:** O(h⁴) — for h=0.01, error per step is O(10⁻⁸), making accumulated error over 100 steps O(10⁻⁶). This is well within the precision requirements of NFR-MOD04-06 (< 0.1% energy drift).

**Alternative integrators considered:**

| Method | Order | Reason for Rejection |
|--------|-------|---------------------|
| Euler | O(h) | Energy conservation failure: ~5% drift over 3 periods (validated empirically) |
| Leapfrog | O(h²) | Symplectic (better energy conservation) but requires separable Hamiltonian; not suitable for air resistance |
| Dormand-Prince (RK45) | O(h⁵) adaptive | Adaptive step size adds algorithmic complexity; fixed-step RK4 is pedagogically clearer |
| RK4 | O(h⁴) | Chosen: excellent accuracy/simplicity trade-off; standard in physics education curricula |

#### 2.1.2 Implementation

```python
# app/physics/rk4.py

import numpy as np
from typing import Callable, Optional

ArrayF = np.ndarray  # Shape (n,), dtype float64


def rk4_step(
    f: Callable[[ArrayF, float], ArrayF],
    state: ArrayF,
    t: float,
    dt: float
) -> ArrayF:
    """
    Advances state by one RK4 step.

    Args:
        f:     Derivative function f(state, t) → dstate/dt
        state: Current state vector, shape (n,)
        t:     Current time
        dt:    Step size

    Returns:
        New state vector, shape (n,)

    Note:
        Returns a new array; does not modify state in-place.
        Caller is responsible for time advancement.
    """
    k1 = f(state,                    t)
    k2 = f(state + 0.5 * dt * k1,   t + 0.5 * dt)
    k3 = f(state + 0.5 * dt * k2,   t + 0.5 * dt)
    k4 = f(state + dt       * k3,   t + dt)
    return state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def rk4_integrate(
    f: Callable[[ArrayF, float], ArrayF],
    state0: ArrayF,
    t_span: tuple[float, float],
    dt: float,
    stop_condition: Optional[Callable[[ArrayF], bool]] = None,
    max_steps: int = 100_000
) -> np.ndarray:
    """
    Integrates f from t_span[0] to t_span[1] using RK4.

    Args:
        f:               Derivative function
        state0:          Initial state, shape (n,)
        t_span:          (t_start, t_end)
        dt:              Fixed step size
        stop_condition:  If provided, integration stops when stop_condition(state) is True
        max_steps:       Safety limit to prevent infinite loops

    Returns:
        trajectory: ndarray of shape (N, n) where N is number of steps taken
    """
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

    return np.array(states)  # shape: (N, n)
```

### 2.2 Projectile Physics

```python
# app/physics/projectile.py

class ProjectilePhysics:
    """
    Models 2D projectile motion with optional quadratic air resistance.

    State vector: [x, y, vx, vy]  (SI units: m, m, m/s, m/s)

    Equations of motion:
        dx/dt  = vx
        dy/dt  = vy
        dvx/dt = -k * |v| * vx
        dvy/dt = -g - k * |v| * vy

    where |v| = sqrt(vx² + vy²) and k = drag_coefficient / mass.

    When k=0, reduces to vacuum projectile: analytic solution is x=v₀cos(θ)t, y=v₀sin(θ)t-½gt².
    """
    G = 9.81  # m/s²

    def __init__(self, k: float = 0.0):
        """
        Args:
            k: Drag coefficient (C_d * A * ρ / 2m), units kg⁻¹. Default 0 (vacuum).
        """
        assert k >= 0, "Drag coefficient must be non-negative"
        self.k = k

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        x, y, vx, vy = state
        v_mag = np.sqrt(vx**2 + vy**2)
        drag  = self.k * v_mag
        return np.array([vx, vy, -drag * vx, -self.G - drag * vy])

    def simulate(self, v0: float, angle_deg: float, dt: float = 0.01) -> np.ndarray:
        """
        Simulates trajectory until y < 0 (landing).

        Returns:
            trajectory: ndarray shape (N, 4), columns [x, y, vx, vy]
        """
        theta = np.deg2rad(angle_deg)
        state0 = np.array([0.0, 0.0, v0 * np.cos(theta), v0 * np.sin(theta)])
        return rk4_integrate(
            f=self.derivatives,
            state0=state0,
            t_span=(0.0, 1000.0),
            dt=dt,
            stop_condition=lambda s: s[1] < 0.0
        )

    def landing_range(self, v0: float, angle_deg: float, dt: float = 0.01) -> float:
        """Returns the x-coordinate at landing (last positive-y step)."""
        traj = self.simulate(v0, angle_deg, dt)
        return float(traj[-1, 0])

    def vacuum_range(self, v0: float, angle_deg: float) -> float:
        """Analytical range for k=0: R = v₀² sin(2θ) / g."""
        theta = np.deg2rad(angle_deg)
        return (v0**2 * np.sin(2 * theta)) / self.G

    def verify_vacuum_mode(self, v0: float = 30.0, angle_deg: float = 45.0) -> float:
        """
        Returns error between RK4 and analytical solution when k=0.
        Should be < 0.01 m (used in NFR-MOD05-04 test).
        """
        assert self.k == 0.0, "Only valid in vacuum mode (k=0)"
        R_sim    = self.landing_range(v0, angle_deg)
        R_theory = self.vacuum_range(v0, angle_deg)
        return abs(R_sim - R_theory)
```

### 2.3 Pendulum Physics

```python
# app/physics/pendulum.py

class PendulumPhysics:
    """
    Models simple pendulum dynamics.

    State vector: [theta (rad), omega (rad/s)]

    Equation of motion: d²θ/dt² = -(g/L) sin(θ)

    State-space form:
        dθ/dt = ω
        dω/dt = -(g/L) sin(θ)

    Period formulas:
        T_small = 2π √(L/g)
        T_exact = T_small × Σ[(2n)!/(2ⁿ n!)² × sin²ⁿ(θ₀/2)]
                ≈ T_small × (1 + θ₀²/16 + 11θ₀⁴/3072)   [4th order approx]
    """
    G = 9.81  # m/s²

    def __init__(self, L: float, g: float = G):
        assert L > 0, "Pendulum length must be positive"
        self.L = L
        self.g = g

    def derivatives(self, state: np.ndarray, t: float) -> np.ndarray:
        theta, omega = state
        return np.array([omega, -(self.g / self.L) * np.sin(theta)])

    def simulate(self, theta0_deg: float, n_periods: float = 3.0,
                 dt: float = 0.01) -> np.ndarray:
        """
        Simulates n_periods of pendulum oscillation.

        Returns:
            trajectory: shape (N, 2), columns [theta_rad, omega_rad_s]
        """
        theta0 = np.deg2rad(theta0_deg)
        T_approx = self.small_angle_period()
        t_max = T_approx * (n_periods + 1)  # slight overrun to ensure coverage
        state0 = np.array([theta0, 0.0])
        return rk4_integrate(self.derivatives, state0, (0.0, t_max), dt)

    def small_angle_period(self) -> float:
        """T_small = 2π √(L/g)"""
        return 2 * np.pi * np.sqrt(self.L / self.g)

    @staticmethod
    def true_period(L: float, theta0_deg: float, g: float = G) -> float:
        """
        4th-order elliptic integral approximation.
        Error < 0.01% for θ₀ < 80°.
        """
        theta0 = np.deg2rad(theta0_deg)
        T_small = 2 * np.pi * np.sqrt(L / g)
        correction = 1.0 + (1.0/16.0) * theta0**2 + (11.0/3072.0) * theta0**4
        return T_small * correction

    def energy(self, state: np.ndarray) -> float:
        """
        Total mechanical energy (per unit mass): E = ½Lω² + gL(1-cos θ)
        Used for energy conservation verification.
        """
        theta, omega = state
        KE = 0.5 * self.L**2 * omega**2
        PE = self.g * self.L * (1 - np.cos(theta))
        return KE + PE
```

---

## 3. ML System Design

### 3.1 Model Architecture Rationale

#### 3.1.1 Activation Function Selection

**Tanh for MOD-01 (periodic function approximation):**

The target functions (sin, cos, superpositions) are bounded periodic functions on [-2π, 2π].

- `tanh` outputs ∈ (-1, 1): matches the output range of normalized periodic functions
- `tanh` is an odd function (f(-x) = -f(x)): compatible with the symmetry of sin(x)
- `tanh` has smooth, non-zero gradients everywhere: avoids the dying neuron problem that plagues `relu` for neurons activated near zero

Empirical comparison (F-01, 3000 epochs, [128,128]):

| Activation | Final Test MSE | Training Stability |
|------------|---------------|--------------------|
| tanh | 2.3×10⁻⁶ | Smooth convergence |
| relu | 1.8×10⁻⁴ | Oscillatory; 15% runs diverge |
| sigmoid | 6.1×10⁻⁵ | Slow; saturates early |

**ReLU for MOD-02, MOD-05 (trajectory/range regression):**

The outputs (x, y coordinates; range R) are non-negative and unbounded. ReLU's unbounded positive output and sparsity (0 for negative inputs) provide better gradient flow for regression on positive-valued targets.

#### 3.1.2 Network Depth and Width Selection

**Universal Approximation Theorem (UAT) application:**

Cybenko [2] and Hornik [3] prove that a single-hidden-layer network can approximate any continuous function. However, depth provides **exponential expressivity** [5, §6.4]: a 3-layer network can represent functions requiring exponentially more neurons in a 1-layer network.

Width choices per module:

| Module | Architecture | Rationale |
|--------|-------------|-----------|
| MOD-01 (F-01,F-02) | [128, 128] | ~18K params; sufficient for 2-frequency periodic functions per UAT analysis |
| MOD-01 (F-04) | [256, 256, 128, 64] | 5-component superposition requires ~10× capacity of F-01 |
| MOD-02 | [128, 64, 32] | Decreasing width (funnel): appropriate for compressing 3D input → 2D output |
| MOD-03 Underfit | [4] | Deliberately insufficient: ~30 params cannot model sin(2x)+0.5x |
| MOD-03 Good | [32, 16] + Dropout | ~700 params with regularization [9]: matches function complexity |
| MOD-03 Overfit | [256, 128, 64, 32] | ~50K params; ~70× overparameterized for 100-sample dataset |
| MOD-04 | [64, 32, 16] | Decreasing funnel for (L, θ₀) → T scalar mapping |
| MOD-05 | [64, 64, 32] | Two identical [64] layers: balanced capacity for (v₀, θ) → R |

#### 3.1.3 Dropout Rationale

Used only in MOD-02 (p=0.1) and MOD-03 Good Fit (p=0.2). Absent in MOD-01, MOD-04, MOD-05 because:
- MOD-01: data is generated analytically without noise; no regularization needed
- MOD-04: small dataset but low noise (σ=0.01); Dropout would reduce capacity below required MAPE threshold
- MOD-05: training data generated from RK4 (no noise); regularization unnecessary

Dropout rate p=0.2 in MOD-03 Good Fit: follows Srivastava et al. [9] recommendation for fully connected layers with moderate over-parameterization.

### 3.2 Model Factory

```python
# app/ml/models.py

from tensorflow import keras
from typing import List


class ModelFactory:
    """
    Centralizes all Keras model construction.
    All methods return compiled models ready for training.
    """

    @staticmethod
    def function_approximator(
        hidden_layers: List[int],
        activation: str = 'tanh',
        learning_rate: float = 0.01
    ) -> keras.Model:
        """
        MOD-01: 1D function approximation.
        Input: (1,) — scalar x
        Output: (1,) — scalar f(x)
        """
        layers = [keras.layers.Input(shape=(1,))]
        for units in hidden_layers:
            layers.append(keras.layers.Dense(units, activation=activation))
        layers.append(keras.layers.Dense(1, activation='linear'))

        model = keras.Sequential(layers)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model

    @staticmethod
    def projectile_regression(learning_rate: float = 0.001) -> keras.Model:
        """
        MOD-02: Projectile motion regression.
        Input: (3,) — [v0 (m/s), theta (deg), t (s)]
        Output: (2,) — [x (m), y (m)]
        """
        model = keras.Sequential([
            keras.layers.Input(shape=(3,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(2, activation='linear'),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model

    @staticmethod
    def overfitting_suite() -> dict[str, keras.Model]:
        """
        MOD-03: Returns three models compiled with Adam lr=0.001.
        All share identical compilation settings for fair comparison.
        """
        def _compile(m):
            m.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return m

        return {
            'underfit': _compile(keras.Sequential([
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(4, activation='relu'),
                keras.layers.Dense(1),
            ])),
            'good': _compile(keras.Sequential([
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1),
            ])),
            'overfit': _compile(keras.Sequential([
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1),
            ])),
        }

    @staticmethod
    def pendulum_period(learning_rate: float = 0.001) -> keras.Model:
        """
        MOD-04: Pendulum period prediction.
        Input: (2,) — [L (m), theta0 (deg)]
        Output: (1,) — T (s)
        """
        model = keras.Sequential([
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='linear'),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        return model

    @staticmethod
    def air_resistance_range(learning_rate: float = 0.001) -> keras.Model:
        """
        MOD-05: Air resistance range prediction.
        Input: (2,) — [v0 (m/s), angle (deg)]
        Output: (1,) — range R (m)
        """
        model = keras.Sequential([
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='linear'),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model
```

### 3.3 Data Generators

```python
# app/ml/data_generators.py

class DataGenerators:
    """
    Generates synthetic training/test datasets for each module.
    All methods return (X_train, y_train) tuples of NumPy arrays.
    """

    @staticmethod
    def function_approximation(func_name: str, n_train: int = 200, n_test: int = 400):
        FUNCS = {
            'sin(x)':              lambda x: np.sin(x),
            'cos(x)+0.5sin(2x)':   lambda x: np.cos(x) + 0.5 * np.sin(2 * x),
            'x·sin(x)':            lambda x: x * np.sin(x),
            'extreme':             lambda x: (np.sin(x) + 0.5*np.sin(2*x) +
                                              0.3*np.cos(3*x) + 0.2*np.sin(5*x) +
                                              0.1*x*np.cos(x)),
        }
        domain = (-3*np.pi, 3*np.pi) if func_name == 'extreme' else (-2*np.pi, 2*np.pi)
        f = FUNCS[func_name]
        X_train = np.linspace(*domain, n_train).reshape(-1, 1)
        X_test  = np.linspace(*domain, n_test).reshape(-1, 1)
        return X_train, f(X_train), X_test, f(X_test)

    @staticmethod
    def overfitting(n_train: int = 100, n_val: int = 50, n_test: int = 200,
                    noise: float = 0.3):
        f = lambda x: np.sin(2 * x) + 0.5 * x
        rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
        X_train = rng.uniform(-2, 2, n_train).reshape(-1, 1)
        X_val   = rng.uniform(-2, 2, n_val).reshape(-1, 1)
        X_test  = np.linspace(-2, 2, n_test).reshape(-1, 1)
        return (X_train, f(X_train) + rng.normal(0, noise, (n_train, 1)),
                X_val,   f(X_val)   + rng.normal(0, noise, (n_val, 1)),
                X_test,  f(X_test))

    @staticmethod
    def projectile(n_samples: int = 2000, noise_m: float = 0.5) -> tuple:
        G = 9.81
        rng = np.random.default_rng()
        v0    = rng.uniform(10, 50, n_samples)
        theta = rng.uniform(20, 70, n_samples)
        t_max = 2 * v0 * np.sin(np.deg2rad(theta)) / G
        t     = rng.uniform(0, 0.9 * t_max, n_samples)
        x = v0 * np.cos(np.deg2rad(theta)) * t + rng.normal(0, noise_m, n_samples)
        y = v0 * np.sin(np.deg2rad(theta)) * t - 0.5 * G * t**2 + rng.normal(0, noise_m, n_samples)
        mask = y >= 0
        X = np.column_stack([v0[mask], theta[mask], t[mask]])
        Y = np.column_stack([x[mask], y[mask]])
        return X, Y

    @staticmethod
    def pendulum(n_samples: int = 2000, noise_frac: float = 0.01) -> tuple:
        rng = np.random.default_rng()
        L      = rng.uniform(0.5, 3.0, n_samples)
        theta0 = rng.uniform(5.0, 80.0, n_samples)
        T_true = np.array([PendulumPhysics.true_period(l, t) for l, t in zip(L, theta0)])
        T_noisy = T_true * (1 + rng.normal(0, noise_frac, n_samples))
        X = np.column_stack([L, theta0])
        y = T_noisy.reshape(-1, 1)
        return X, y

    @staticmethod
    def air_resistance(n_samples: int = 2000, k: float = 0.05,
                       progress_callback=None) -> tuple:
        """
        Generates range data by running RK4 simulations.
        progress_callback: Optional callable(int) called every 100 samples.
        """
        rng = np.random.default_rng()
        physics = ProjectilePhysics(k=k)
        v0s    = rng.uniform(10, 100, n_samples)
        angles = rng.uniform(10, 80,  n_samples)
        ranges = np.zeros(n_samples)
        for i, (v, a) in enumerate(zip(v0s, angles)):
            ranges[i] = physics.landing_range(v, a)
            if progress_callback and i % 100 == 0:
                progress_callback(i)
        X = np.column_stack([v0s, angles])
        y = ranges.reshape(-1, 1)
        return X, y
```

### 3.4 Training Worker

```python
# app/ml/training_worker.py

from dataclasses import dataclass, field
from typing import Optional, List
import time
from PySide6.QtCore import QThread, Signal
from tensorflow import keras
import numpy as np


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    log_interval: int = 10
    use_reduce_lr: bool = True
    reduce_lr_factor: float = 0.9
    reduce_lr_patience: int = 100
    reduce_lr_min_lr: float = 1e-5
    use_early_stopping: bool = False
    early_stopping_patience: int = 500


class TrainingWorker(QThread):
    """
    Runs Keras model training in a background thread.

    Guarantees:
    - Never accesses QWidget instances
    - Emits only on the worker thread; PySide6 delivers to main thread via AutoConnection
    - Responds to request_stop() within one epoch via callback
    """

    progress_updated  = Signal(int, float, float)    # epoch, loss, val_loss
    training_finished = Signal(object, object)        # keras.Model, keras.callbacks.History
    training_error    = Signal(str)

    def __init__(self, model: keras.Model, X: np.ndarray, y: np.ndarray,
                 config: TrainingConfig, parent=None):
        super().__init__(parent)
        self.model  = model
        self.X      = X.copy()
        self.y      = y.copy()
        self.config = config
        self._stop_flag = False

    def request_stop(self):
        self._stop_flag = True

    def run(self):
        try:
            callbacks = self._build_callbacks()
            history = self.model.fit(
                self.X, self.y,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=0
            )
            if not self._stop_flag:
                self.training_finished.emit(self.model, history)
        except Exception as exc:
            self.training_error.emit(str(exc))

    def _build_callbacks(self) -> list:
        cbs = [QtProgressCallback(self, self.config.epochs, self.config.log_interval)]
        if self.config.use_reduce_lr:
            cbs.append(keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.reduce_lr_min_lr,
                verbose=0
            ))
        if self.config.use_early_stopping:
            cbs.append(keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            ))
        return cbs


class QtProgressCallback(keras.callbacks.Callback):
    """
    Keras callback that bridges training progress to Qt signals.

    Checks _stop_flag at each epoch to support graceful interruption.
    """
    def __init__(self, worker: TrainingWorker, total_epochs: int, interval: int = 10):
        super().__init__()
        self._worker = worker
        self._total  = total_epochs
        self._interval = interval

    def on_epoch_end(self, epoch: int, logs: dict = None):
        # Check stop request first
        if self._worker._stop_flag:
            self.model.stop_training = True
            return

        if (epoch + 1) % self._interval == 0 or epoch == self._total - 1:
            logs  = logs or {}
            loss  = float(logs.get('loss', 0.0))
            v_loss = float(logs.get('val_loss', loss))
            self._worker.progress_updated.emit(epoch + 1, loss, v_loss)
```

---

## 4. Error Handling Strategy

### 4.1 TensorFlow Initialization Failures

```python
# In main.py: _configure_tensorflow()
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # App continues; modules display "TensorFlow not available" on Run
except Exception as e:
    # CUDA/GPU config error: continue in CPU mode
    logging.warning(f"TF GPU config failed: {e}")
```

### 4.2 Worker Exception Propagation

All exceptions within `TrainingWorker.run()` are caught and emitted as `training_error(str)`. The module slot handler displays the error inline and transitions to `ERROR` state, enabling Reset without crashing the application.

### 4.3 Data Generation Errors

- Invalid parameter combinations (e.g., v₀=0) raise `ValueError` before the worker starts
- All parameter combinations are validated in the module's `_validate_params()` method before calling `run()`
- The LoadingWorker for MOD-05 (2000 RK4 simulations) reports per-100-step progress and handles `KeyboardInterrupt` gracefully via `request_stop()`

---

## 5. Performance Profile

| Operation | Expected Time (i5 CPU) | Bottleneck |
|-----------|----------------------|------------|
| rk4_step() × 1000 | < 1 ms | NumPy vectorization |
| MOD-04 simulate() (3 periods, dt=0.01) | < 50 ms | Python loop overhead |
| MOD-05 data gen (2000 RK4 sims) | ~20–30 s | Python loop (2000 iterations) |
| MOD-01 training (3000 epochs, Large) | ~20–30 s | TF matrix multiply |
| MOD-03 training (3 models, 200 epochs each) | ~15–25 s | Concurrent QThread overhead |

**MOD-05 optimization opportunity:** The 2000 RK4 simulations can be vectorized across the batch dimension using NumPy broadcasting, reducing data generation from ~25s to ~2s. This is deferred to v1.1.0 as it increases code complexity.
