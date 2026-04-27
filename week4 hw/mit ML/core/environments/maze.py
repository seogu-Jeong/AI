from __future__ import annotations
import numpy as np
from collections import deque


class Maze:
    """Recursive-backtracking perfect maze.

    walls[y, x, d] = True  means wall exists in direction d.
    Directions: 0=N, 1=S, 2=W, 3=E
    States: row * cols + col
    Actions: 0=N, 1=S, 2=W, 3=E
    """

    DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]   # (dx, dy) for N,S,W,E
    OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

    def __init__(self, rows: int = 5, cols: int = 5, seed: int | None = None):
        self.rows = rows
        self.cols = cols
        self.n_states = rows * cols
        self.n_actions = 4
        # walls[y, x, direction]: True = wall present
        self.walls = np.ones((rows, cols, 4), dtype=bool)
        rng = np.random.default_rng(seed)
        self._generate(rng)
        self._agent_row = 0
        self._agent_col = 0

    def _generate(self, rng: np.random.Generator):
        """Iterative DFS (recursive backtracking) maze generation."""
        visited = np.zeros((self.rows, self.cols), dtype=bool)
        stack = [(0, 0)]
        visited[0, 0] = True

        while stack:
            y, x = stack[-1]
            dirs = list(range(4))
            rng.shuffle(dirs)
            moved = False
            for d in dirs:
                dx, dy = self.DIRS[d]
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows and not visited[ny, nx]:
                    # Remove wall between (y,x) and (ny,nx)
                    self.walls[y, x, d] = False
                    self.walls[ny, nx, self.OPPOSITE[d]] = False
                    visited[ny, nx] = True
                    stack.append((ny, nx))
                    moved = True
                    break
            if not moved:
                stack.pop()

    def reset(self) -> tuple[int, int]:
        self._agent_row = 0
        self._agent_col = 0
        return (0, 0)

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        dx, dy = self.DIRS[action]
        r, c = self._agent_row, self._agent_col

        # Check wall
        if not self.walls[r, c, action]:
            nr, nc = r + dy, c + dx
            self._agent_row, self._agent_col = nr, nc
        else:
            nr, nc = r, c

        done = (nr == self.rows - 1 and nc == self.cols - 1)
        reward = 1.0 if done else -0.01
        return (nr, nc), reward, done

    def bfs_shortest(self) -> int:
        """Return length of shortest path from (0,0) to goal."""
        return len(self.bfs_path()) - 1

    def bfs_path(self) -> list[tuple[int, int]]:
        """Return BFS shortest path as list of (row, col) tuples."""
        start = (0, 0); goal = (self.rows - 1, self.cols - 1)
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == goal:
                return path
            for d in range(4):
                if not self.walls[r, c, d]:
                    dx, dy = self.DIRS[d]
                    nr, nc = r + dy, c + dx
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [(nr, nc)]))
        return [start]
