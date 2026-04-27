from __future__ import annotations
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer

from core.environments.gridworld import GridWorld
from core.algorithms.qlearning import QLearning
from core.algorithms.reinforce import REINFORCE, LinearPolicy


class ArenaController(QObject):
    """Runs Q-Learning vs REINFORCE head-to-head on GridWorld."""

    step_done = Signal(int, float, float)    # (episode, ql_reward, rf_reward)
    q_table_updated = Signal(object)         # QLearning Q-table (np.ndarray)
    rf_policy_updated = Signal(object)       # REINFORCE policy weights (np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)

        n = 5
        self._env_ql = GridWorld(n=n)
        self._env_rf = GridWorld(n=n)
        n_states = n * n
        self._ql = QLearning(n_states=n_states, n_actions=4,
                             alpha=0.1, gamma=0.95, epsilon=0.2)
        self._rf = REINFORCE(policy=LinearPolicy(n_states, 4),
                             gamma=0.95, lr=0.01)
        self._episode = 0
        self._ql_state = self._env_ql.reset()
        self._rf_state = self._env_rf.reset()
        self._ql_ep_reward = 0.0; self._rf_ep_reward = 0.0
        self._ql_steps = 0; self._rf_steps = 0
        self._rf_ep_states: list = []; self._rf_ep_actions: list = []
        self._rf_ep_rewards: list = []

    def start(self):
        self._timer.start(50)

    def stop(self):
        self._timer.stop()

    def reset(self):
        self.stop()
        self._episode = 0
        self._env_ql = GridWorld(n=5)
        self._env_rf = GridWorld(n=5)
        n_states = 25
        self._ql = QLearning(n_states=n_states, n_actions=4,
                             alpha=0.1, gamma=0.95, epsilon=0.2)
        self._rf = REINFORCE(policy=LinearPolicy(n_states, 4),
                             gamma=0.95, lr=0.01)
        self._ql_state = self._env_ql.reset()
        self._rf_state = self._env_rf.reset()
        self._ql_ep_reward = 0.0; self._rf_ep_reward = 0.0
        self._ql_steps = 0; self._rf_steps = 0
        self._rf_ep_states = []; self._rf_ep_actions = []; self._rf_ep_rewards = []

    def _step(self):
        # Q-Learning step
        a_ql = self._ql.select_action(self._ql_state)
        ns_ql, r_ql, done_ql = self._env_ql.step(a_ql)
        self._ql.update(self._ql_state, a_ql, r_ql, ns_ql, done_ql)
        self._ql_ep_reward += r_ql
        self._ql_state = ns_ql; self._ql_steps += 1
        ql_done = done_ql or self._ql_steps > 200

        # REINFORCE step
        a_rf = self._rf.select_action(self._rf_state)
        ns_rf, r_rf, done_rf = self._env_rf.step(a_rf)
        self._rf_ep_states.append(self._rf_state)
        self._rf_ep_actions.append(a_rf)
        self._rf_ep_rewards.append(r_rf)
        self._rf_ep_reward += r_rf
        self._rf_state = ns_rf; self._rf_steps += 1
        rf_done = done_rf or self._rf_steps > 200

        if ql_done and rf_done:
            self._rf.finish_episode(
                self._rf_ep_states, self._rf_ep_actions, self._rf_ep_rewards)
            self._episode += 1
            self.step_done.emit(self._episode, self._ql_ep_reward, self._rf_ep_reward)
            self.q_table_updated.emit(self._ql.q_table.copy())
            self.rf_policy_updated.emit(self._rf.policy.W.copy())
            self._ql_state = self._env_ql.reset()
            self._rf_state = self._env_rf.reset()
            self._ql_ep_reward = 0.0; self._rf_ep_reward = 0.0
            self._ql_steps = 0; self._rf_steps = 0
            self._rf_ep_states = []; self._rf_ep_actions = []; self._rf_ep_rewards = []
