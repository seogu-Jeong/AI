from __future__ import annotations
import numpy as np


class GridWorld:
    """N×N grid MDP with obstacles, cliff cells, and a goal state.

    States:  row * n + col  (int, 0 … n²-1)
    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    """

    ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(
        self,
        n: int = 4,
        obstacles: list[int] | None = None,
        cliffs: list[int] | None = None,
        goal_state: int | None = None,
    ):
        self.n = n
        self.obstacles: set[int] = set(obstacles or [])
        self.cliffs: set[int] = set(cliffs or [])
        self.goal_state: int = goal_state if goal_state is not None else n * n - 1
        self.agent_state: int = 0
        self.n_states: int = n * n
        self.n_actions: int = 4

    def reset(self) -> int:
        self.agent_state = 0
        return self.agent_state

    def step(self, action: int) -> tuple[int, float, bool]:
        dr, dc = self.ACTION_DELTAS[action]
        r, c = self.agent_state // self.n, self.agent_state % self.n
        nr, nc = r + dr, c + dc

        # Wall bounce
        if not (0 <= nr < self.n and 0 <= nc < self.n):
            nr, nc = r, c

        next_state = nr * self.n + nc

        # Obstacle bounce
        if next_state in self.obstacles:
            next_state = self.agent_state

        self.agent_state = next_state

        # Rewards
        if next_state == self.goal_state:
            return next_state, 1.0, True
        if next_state in self.cliffs:
            self.agent_state = 0  # reset to start
            return 0, -10.0, False

        return next_state, -0.01, False
