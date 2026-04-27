from __future__ import annotations
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer

from core.environments.gridworld import GridWorld
from core.environments.cartpole import CartPole
from core.environments.maze import Maze
from core.algorithms.qlearning import QLearning
from core.algorithms.reinforce import REINFORCE, LinearPolicy


class TrainingController(QObject):
    """Drives training loop via QTimer. Emits signals consumed by views."""

    episode_done = Signal(int, float)
    agent_moved = Signal(int, float)           # (state, reward) — GridWorld
    maze_moved = Signal(tuple, float)          # ((r,c), reward) — Maze
    cartpole_state = Signal(float, float, float, float, bool)  # x,xd,th,thd,done

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)
        self._env = None
        self._agent = None
        self._env_type = "gridworld"
        self._episode = 0
        self._steps = 0
        self._states: list = []
        self._actions: list = []
        self._rewards: list = []
        self._episode_reward = 0.0
        self._speed_ms = 100
        self._current_state = 0

    def setup(self, env_type: str, env_cfg: dict, algo_type: str, algo_cfg: dict):
        self._env_type = env_type
        if env_type == "gridworld":
            self._env = GridWorld(**env_cfg)
            n_states = self._env.n * self._env.n
            if algo_type == "qlearning":
                self._agent = QLearning(n_states=n_states, n_actions=4, **algo_cfg)
            else:
                self._agent = REINFORCE(LinearPolicy(n_states, 4), **algo_cfg)
        elif env_type == "cartpole":
            self._env = CartPole(**env_cfg)
            n_states = 10 ** 4
            if algo_type == "qlearning":
                self._agent = QLearning(n_states=n_states, n_actions=2, **algo_cfg)
            else:
                self._agent = REINFORCE(LinearPolicy(n_states, 2), **algo_cfg)
        elif env_type == "maze":
            self._env = Maze(**env_cfg)
            n_states = self._env.rows * self._env.cols
            if algo_type == "qlearning":
                self._agent = QLearning(n_states=n_states, n_actions=4, **algo_cfg)
            else:
                self._agent = REINFORCE(LinearPolicy(n_states, 4), **algo_cfg)
        self._reset_episode()

    def set_speed(self, steps_per_sec: int):
        self._speed_ms = max(10, 1000 // max(1, steps_per_sec))
        if self._timer.isActive():
            self._timer.setInterval(self._speed_ms)

    def start(self):
        if self._env and self._agent:
            self._timer.start(self._speed_ms)

    def stop(self):
        self._timer.stop()

    def reset(self):
        self.stop()
        if self._env: self._reset_episode()
        self._episode = 0

    def get_q_table(self) -> np.ndarray | None:
        if hasattr(self._agent, 'q_table'):
            return self._agent.q_table
        return None

    @property
    def env(self):
        return self._env

    @property
    def agent(self):
        return self._agent

    def _reset_episode(self):
        if self._env is None: return
        raw = self._env.reset()
        if self._env_type == "cartpole":
            self._current_state = self._env.discretize(raw)
        elif self._env_type == "maze":
            r, c = raw
            self._current_state = r * self._env.cols + c
        else:
            self._current_state = raw
        self._states = []; self._actions = []; self._rewards = []
        self._episode_reward = 0.0
        self._steps = 0

    def _step(self):
        if self._env is None or self._agent is None: return
        s = self._current_state
        a = self._agent.select_action(s)

        if self._env_type == "maze":
            (r, c), reward, done = self._env.step(a)
            ns = r * self._env.cols + c
            self.maze_moved.emit((r, c), reward)
        elif self._env_type == "cartpole":
            obs, reward, done = self._env.step(a)
            ns = self._env.discretize(obs)
            self.cartpole_state.emit(obs[0], obs[1], obs[2], obs[3], done)
        else:
            ns, reward, done = self._env.step(a)
            self.agent_moved.emit(ns, reward)

        if hasattr(self._agent, 'update'):
            self._agent.update(s, a, reward, ns, done)

        self._states.append(s); self._actions.append(a); self._rewards.append(reward)
        self._episode_reward += reward
        self._current_state = ns
        self._steps += 1

        if done or self._steps > 500:
            if hasattr(self._agent, 'finish_episode'):
                self._agent.finish_episode(self._states, self._actions, self._rewards)
            self._episode += 1
            self.episode_done.emit(self._episode, self._episode_reward)
            self._reset_episode()
