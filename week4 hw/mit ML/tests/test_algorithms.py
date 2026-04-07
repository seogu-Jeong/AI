import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from core.algorithms.qlearning import QLearning
from core.algorithms.reinforce import REINFORCE, LinearPolicy
from core.environments.gridworld import GridWorld


def test_qlearning_update_changes_q():
    ql = QLearning(n_states=4, n_actions=4, alpha=0.5, gamma=0.9, epsilon=0.0)
    ql.update(state=0, action=1, reward=1.0, next_state=3, done=True)
    assert ql.q_table[0, 1] == pytest.approx(0.5, abs=1e-6)


def test_qlearning_greedy_selects_max():
    ql = QLearning(n_states=4, n_actions=4, epsilon=0.0)
    ql.q_table[0, 2] = 5.0
    assert ql.select_action(0) == 2


def test_qlearning_run_episode_converges():
    np.random.seed(0)
    env = GridWorld(n=3)
    ql = QLearning(n_states=9, n_actions=4, alpha=0.3, gamma=0.9, epsilon=0.3)
    rewards = [ql.run_episode(env)[0] for _ in range(300)]
    # Later episodes should have higher mean reward than early ones
    assert np.mean(rewards[-50:]) > np.mean(rewards[:50])


def test_reinforce_policy_forward_sums_to_one():
    pol = LinearPolicy(n_states=4, n_actions=4)
    probs = pol.forward(state=0)
    assert abs(np.sum(probs) - 1.0) < 1e-6
    assert len(probs) == 4


def test_reinforce_finish_episode_updates_weights():
    np.random.seed(42)
    env = GridWorld(n=2)
    pol = LinearPolicy(n_states=4, n_actions=4)
    rf = REINFORCE(policy=pol, gamma=0.9, lr=0.01)
    W_before = pol.W.copy()
    states, actions, rewards = [0], [3], [1.0]
    rf.finish_episode(states, actions, rewards)
    assert not np.allclose(pol.W, W_before)
