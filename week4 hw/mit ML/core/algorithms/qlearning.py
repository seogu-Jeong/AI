from __future__ import annotations
import numpy as np


class QLearning:
    """Tabular Q-Learning with ε-greedy action selection."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state: int) -> int:
        """ε-greedy: random with prob ε, else argmax Q."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool) -> None:
        """TD update: Q(s,a) ← Q(s,a) + α(r + γ·max Q(s',·) - Q(s,a))"""
        target = reward + self.gamma * float(np.max(self.q_table[next_state])) * (1 - int(done))
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

    def run_episode(self, env) -> tuple[float, int, list[int]]:
        """Run one full episode. Returns (total_reward, steps, trajectory)."""
        state = env.reset()
        total_reward = 0.0
        steps = 0
        trajectory: list[int] = [state]
        done = False
        while not done and steps < 1000:
            action = self.select_action(state)
            next_state, reward, done = env.step(action)
            self.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            trajectory.append(state)
        return total_reward, steps, trajectory
