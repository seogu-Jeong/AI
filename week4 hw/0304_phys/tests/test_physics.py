import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from app.physics.rk4 import rk4_step, rk4_integrate


def test_rk4_step_exponential():
    # dy/dt = -y  → y(t) = exp(-t); one step from y=1, dt=0.1 → ~0.9048
    f = lambda s, t: np.array([-s[0]])
    s0 = np.array([1.0])
    s1 = rk4_step(f, s0, 0.0, 0.1)
    assert abs(s1[0] - np.exp(-0.1)) < 1e-6


def test_rk4_integrate_shape():
    f = lambda s, t: np.array([-s[0]])
    traj = rk4_integrate(f, np.array([1.0]), (0.0, 1.0), dt=0.1)
    assert traj.shape[1] == 1
    assert traj.shape[0] > 5


def test_rk4_integrate_stop_condition():
    # Stop when state[0] < 0.5
    f = lambda s, t: np.array([-s[0]])
    traj = rk4_integrate(f, np.array([1.0]), (0.0, 100.0), dt=0.1,
                          stop_condition=lambda s: s[0] < 0.5)
    # Should stop before t=100
    assert traj.shape[0] < 1000
    # Last state should be near 0.5
    assert traj[-1, 0] < 0.6


from app.physics.projectile import ProjectilePhysics


def test_projectile_vacuum_range_accuracy():
    # NFR-MOD05-04: k=0 RK4 must match analytical within 0.01 m
    phys = ProjectilePhysics(k=0.0)
    for v0, angle in [(30, 45), (50, 30), (20, 60)]:
        R_rk4 = phys.landing_range(v0, angle)
        R_theory = phys.vacuum_range(v0, angle)
        assert abs(R_rk4 - R_theory) < 0.01, f"v0={v0}, θ={angle}: err={abs(R_rk4-R_theory):.4f}m"


def test_projectile_drag_reduces_range():
    # With drag, range must be shorter than vacuum
    phys_vac  = ProjectilePhysics(k=0.0)
    phys_drag = ProjectilePhysics(k=0.05)
    R_vac  = phys_vac.landing_range(50, 45)
    R_drag = phys_drag.landing_range(50, 45)
    assert R_drag < R_vac


def test_projectile_simulate_shape():
    phys = ProjectilePhysics(k=0.0)
    traj = phys.simulate(30, 45)
    assert traj.shape[1] == 4   # [x, y, vx, vy]
    assert traj.shape[0] > 10


from app.physics.pendulum import PendulumPhysics


def test_pendulum_small_angle_accuracy():
    # NFR-MOD04-07: at θ₀=5°, small-angle error < 0.1%
    L = 1.0
    phys = PendulumPhysics(L)
    T_small = phys.small_angle_period()
    T_exact = PendulumPhysics.true_period(L, 5.0)
    err_pct = abs(T_small - T_exact) / T_exact * 100
    assert err_pct < 0.1, f"Small-angle error at 5°: {err_pct:.3f}%"


def test_pendulum_large_angle_error():
    # NFR-MOD04-07: at θ₀=60°, small-angle error > 5%
    L = 1.0
    phys = PendulumPhysics(L)
    T_small = phys.small_angle_period()
    T_exact = PendulumPhysics.true_period(L, 60.0)
    err_pct = abs(T_small - T_exact) / T_exact * 100
    assert err_pct > 5.0, f"Expected > 5% error at 60°, got {err_pct:.3f}%"


def test_pendulum_energy_conservation():
    # NFR-MOD04-06: energy drift < 0.1% over 3 periods
    L = 1.0; theta0_deg = 30.0
    phys  = PendulumPhysics(L)
    traj  = phys.simulate(theta0_deg, n_periods=3, dt=0.01)
    E0    = phys.energy(traj[0])
    E_end = phys.energy(traj[-1])
    drift = abs(E_end - E0) / abs(E0) * 100
    assert drift < 0.1, f"Energy drift: {drift:.4f}%"


def test_pendulum_simulate_shape():
    phys = PendulumPhysics(1.0)
    traj = phys.simulate(30.0)
    assert traj.shape[1] == 2   # [theta_rad, omega_rad_s]
