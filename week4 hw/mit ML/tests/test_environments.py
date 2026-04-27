import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from core.environments.gridworld import GridWorld


def test_gridworld_reset_returns_start():
    env = GridWorld(n=4)
    s = env.reset()
    assert s == 0


def test_gridworld_step_goal():
    env = GridWorld(n=2)  # 2x2: states 0,1,2,3; goal=3
    # state=2 (row1,col0), action=Right(3) → state=3 (goal)
    env.reset()
    env.agent_state = 2
    next_s, reward, done = env.step(3)
    assert next_s == 3
    assert reward == 1.0
    assert done is True


def test_gridworld_step_wall_stays():
    env = GridWorld(n=4)
    env.reset()
    env.agent_state = 0
    # Move Up from top-left → stays at 0
    next_s, reward, done = env.step(0)
    assert next_s == 0


def test_gridworld_obstacle_blocks():
    env = GridWorld(n=4, obstacles=[1])
    env.reset()
    env.agent_state = 0
    # Move Right from 0 → would be 1 (obstacle) → stays at 0
    next_s, reward, done = env.step(3)
    assert next_s == 0


def test_gridworld_step_returns_negative_reward():
    env = GridWorld(n=4)
    env.reset()
    next_s, reward, done = env.step(3)
    assert reward == -0.01
    assert done is False
