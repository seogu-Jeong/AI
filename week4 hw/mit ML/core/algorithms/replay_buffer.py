from __future__ import annotations
import numpy as np


class ReplayBuffer:
    """Circular experience replay buffer for DQN concept visualization."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._states = np.zeros(capacity, dtype=np.int64)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros(capacity, dtype=np.int64)
        self._dones = np.zeros(capacity, dtype=bool)
        self._ptr = 0
        self._size = 0

    def push(self, state: int, action: int, reward: float,
             next_state: int, done: bool) -> None:
        self._states[self._ptr] = state
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._next_states[self._ptr] = next_state
        self._dones[self._ptr] = done
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray, np.ndarray]:
        if self._size < batch_size:
            raise ValueError(
                f"Not enough samples: {self._size} < {batch_size}"
            )
        idx = np.random.choice(self._size, size=batch_size, replace=False)
        return (
            self._states[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_states[idx],
            self._dones[idx],
        )

    def __len__(self) -> int:
        return self._size

    @property
    def fill_ratio(self) -> float:
        return self._size / self.capacity
