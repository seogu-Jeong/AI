# RL AI Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PySide6 desktop RL educational dashboard with 3 environments (GridWorld, CartPole, Maze), 3 algorithms (Q-Learning, REINFORCE, DQN-concept), live QPainter visualizations, and a Glassmorphism dark theme — covering all lecture RL concepts interactively.

**Architecture:** Pure-NumPy Core (environments + algorithms) ↔ Controller layer (Qt Signal/Slot bridge) ↔ QPainter UI (visualizations + concept widgets). QTimer-driven training (variable speed) decoupled from 30 FPS render timer. No Matplotlib. Glassmorphism theme via `ui/theme.py` token system.

**Tech Stack:** Python 3.10+, PySide6 ≥ 6.6, NumPy ≥ 1.26. No other dependencies.

---

## File Map

```
rl0407/
├── main.py
├── requirements.txt
├── core/
│   ├── __init__.py
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── gridworld.py
│   │   ├── cartpole.py
│   │   └── maze.py
│   └── algorithms/
│       ├── __init__.py
│       ├── qlearning.py
│       ├── reinforce.py
│       └── replay_buffer.py
├── controllers/
│   ├── __init__.py
│   ├── training_controller.py
│   ├── concept_controller.py
│   └── arena_controller.py
├── ui/
│   ├── __init__.py
│   ├── theme.py
│   ├── main_window.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── glass_panel.py
│   │   ├── sidebar.py
│   │   ├── slider_group.py
│   │   ├── learning_curve.py
│   │   └── status_bar.py
│   ├── visualizations/
│   │   ├── __init__.py
│   │   ├── painter_utils.py
│   │   ├── gridworld_view.py
│   │   ├── cartpole_view.py
│   │   └── maze_view.py
│   └── views/
│       ├── __init__.py
│       ├── concepts/
│       │   ├── __init__.py
│       │   ├── rl_basics_view.py
│       │   ├── value_based_view.py
│       │   ├── policy_based_view.py
│       │   └── applications_view.py
│       ├── playground_view.py
│       └── arena_view.py
└── tests/
    ├── __init__.py
    ├── test_environments.py
    ├── test_algorithms.py
    └── test_replay_buffer.py
```

---

## Task 1: Project Scaffold

**Files:**
- Create: all `__init__.py` files, `requirements.txt`

- [ ] **Step 1: Create directory tree**

```bash
cd /Users/jsw/20260406/rl0407
mkdir -p core/environments core/algorithms controllers \
         ui/components ui/visualizations ui/views/concepts \
         tests docs/superpowers/plans
touch core/__init__.py core/environments/__init__.py core/algorithms/__init__.py \
      controllers/__init__.py \
      ui/__init__.py ui/components/__init__.py ui/visualizations/__init__.py \
      ui/views/__init__.py ui/views/concepts/__init__.py \
      tests/__init__.py
```

- [ ] **Step 2: Create requirements.txt**

`rl0407/requirements.txt`:
```
PySide6>=6.6.0,<7.0.0
numpy>=1.26.0,<2.0.0
```

- [ ] **Step 3: Commit**
```bash
cd /Users/jsw/20260406/rl0407
git add . && git commit -m "chore: project scaffold"
```

---

## Task 2: GridWorld Environment + Tests

**Files:**
- Create: `core/environments/gridworld.py`
- Create: `tests/test_environments.py` (GridWorld tests)

- [ ] **Step 1: Write failing tests**

`tests/test_environments.py`:
```python
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
    # Move Right from 0 → obstacle at 1 → stays at 0
    next_s, reward, done = env.step(3)
    assert next_s == 0


def test_gridworld_n_states_actions():
    env = GridWorld(n=6)
    assert env.n_states == 36
    assert env.n_actions == 4


def test_gridworld_state_to_rc():
    env = GridWorld(n=4)
    r, c = env.state_to_rc(5)  # row=1, col=1
    assert (r, c) == (1, 1)
```

- [ ] **Step 2: Run — expect FAIL**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/test_environments.py -v 2>&1 | head -20
```

- [ ] **Step 3: Implement gridworld.py**

`core/environments/gridworld.py`:
```python
"""
GridWorld — N×N tabular MDP environment. Zero Qt imports.
SRS-RLAI-001 FR-PG-GW-01
"""
import numpy as np


class GridWorld:
    # Actions: 0=Up, 1=Down, 2=Left, 3=Right
    ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    N_ACTIONS = 4

    def __init__(self, n: int = 6, obstacles: list = None, cliff_cells: list = None):
        self.n = n
        self.n_states = n * n
        self.n_actions = self.N_ACTIONS
        self.start_state = 0
        self.goal_state = n * n - 1
        self.obstacles = set(obstacles or [])
        self.cliff_cells = set(cliff_cells or [])
        self.agent_state = self.start_state

    def reset(self) -> int:
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action: int) -> tuple:
        """Returns (next_state, reward, done)."""
        dr, dc = self.ACTION_DELTAS[action]
        r, c = self.state_to_rc(self.agent_state)
        nr, nc = r + dr, c + dc
        # Wall bounce
        if not (0 <= nr < self.n and 0 <= nc < self.n):
            nr, nc = r, c
        next_state = self.rc_to_state(nr, nc)
        # Obstacle bounce
        if next_state in self.obstacles:
            next_state = self.agent_state
        self.agent_state = next_state

        if next_state == self.goal_state:
            return next_state, 1.0, True
        if next_state in self.cliff_cells:
            return next_state, -10.0, True
        return next_state, -0.01, False

    def state_to_rc(self, state: int) -> tuple:
        return divmod(state, self.n)

    def rc_to_state(self, row: int, col: int) -> int:
        return row * self.n + col
```

- [ ] **Step 4: Run — expect PASS**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/test_environments.py -v
```

- [ ] **Step 5: Commit**
```bash
git add core/environments/gridworld.py tests/test_environments.py
git commit -m "feat: GridWorld MDP environment (FR-PG-GW-01)"
```

---

## Task 3: CartPole Environment + Tests

**Files:**
- Create: `core/environments/cartpole.py`
- Modify: `tests/test_environments.py` (append CartPole tests)

- [ ] **Step 1: Append CartPole tests**

Append to `tests/test_environments.py`:
```python
from core.environments.cartpole import CartPole


def test_cartpole_reset_shape():
    env = CartPole()
    state = env.reset()
    assert state.shape == (4,)


def test_cartpole_step_returns_tuple():
    env = CartPole()
    state = env.reset()
    next_state, reward, done = env.step(0)
    assert next_state.shape == (4,)
    assert reward == 1.0
    assert isinstance(done, bool)


def test_cartpole_terminal_angle():
    env = CartPole()
    env.reset()
    # Force extreme angle
    env.state = np.array([0.0, 0.0, 0.25, 0.0])  # theta = 0.25 > 0.2094
    _, _, done = env.step(0)
    assert done is True


def test_cartpole_discretize_returns_int():
    env = CartPole()
    state = env.reset()
    idx = env.discretize(state, bins=10)
    assert isinstance(idx, int)
    assert 0 <= idx < 10**4
```

- [ ] **Step 2: Run CartPole tests — expect FAIL**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/test_environments.py -k "cartpole" -v 2>&1 | head -20
```

- [ ] **Step 3: Implement cartpole.py**

`core/environments/cartpole.py`:
```python
"""
CartPole — Newtonian physics simulation. Zero Qt imports.
SRS-RLAI-001 FR-PG-CP-01
"""
import numpy as np


class CartPole:
    GRAVITY       = 9.8
    MASSCART      = 1.0
    MASSPOLE      = 0.1
    TOTAL_MASS    = 1.1
    LENGTH        = 0.5          # half-pole length
    POLEMASS_LEN  = 0.05         # MASSPOLE * LENGTH
    FORCE_MAG     = 10.0
    TAU           = 0.02         # seconds per step
    THETA_LIMIT   = 0.2094       # 12 degrees in radians
    X_LIMIT       = 2.4

    def __init__(self):
        self.state: np.ndarray = None

    def reset(self) -> np.ndarray:
        """Returns state [x, x_dot, theta, theta_dot] — small random init."""
        self.state = np.random.uniform(-0.05, 0.05, size=(4,))
        return self.state.copy()

    def step(self, action: int) -> tuple:
        """action ∈ {0: push left, 1: push right}. Returns (state, reward, done)."""
        assert action in (0, 1)
        x, x_dot, theta, theta_dot = self.state
        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        temp = (force + self.POLEMASS_LEN * theta_dot**2 * sin_theta) / self.TOTAL_MASS
        thetaacc = (self.GRAVITY * sin_theta - cos_theta * temp) / (
            self.LENGTH * (4.0 / 3.0 - self.MASSPOLE * cos_theta**2 / self.TOTAL_MASS)
        )
        xacc = temp - self.POLEMASS_LEN * thetaacc * cos_theta / self.TOTAL_MASS

        # Euler integration
        x         += self.TAU * x_dot
        x_dot     += self.TAU * xacc
        theta     += self.TAU * theta_dot
        theta_dot += self.TAU * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])

        done = bool(abs(x) > self.X_LIMIT or abs(theta) > self.THETA_LIMIT)
        return self.state.copy(), 1.0, done

    def discretize(self, state: np.ndarray, bins: int = 10) -> int:
        """Maps continuous state to discrete integer index."""
        limits = [
            (-self.X_LIMIT,     self.X_LIMIT),
            (-3.0,              3.0),
            (-self.THETA_LIMIT, self.THETA_LIMIT),
            (-3.0,              3.0),
        ]
        idx = 0
        for i, (s, (lo, hi)) in enumerate(zip(state, limits)):
            b = int(np.clip((s - lo) / (hi - lo) * bins, 0, bins - 1))
            idx += b * (bins ** i)
        return idx
```

- [ ] **Step 4: Run all environment tests — expect PASS**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/test_environments.py -v
```

- [ ] **Step 5: Commit**
```bash
git add core/environments/cartpole.py tests/test_environments.py
git commit -m "feat: CartPole Newtonian physics environment (FR-PG-CP-01)"
```

---

## Task 4: Maze Environment + Tests

**Files:**
- Create: `core/environments/maze.py`
- Modify: `tests/test_environments.py` (append Maze tests)

- [ ] **Step 1: Append Maze tests**

Append to `tests/test_environments.py`:
```python
from core.environments.maze import Maze


def test_maze_reset_returns_start():
    env = Maze(width=7, height=7, seed=0)
    s = env.reset()
    assert s == 0


def test_maze_goal_is_last_state():
    env = Maze(width=7, height=7, seed=0)
    assert env.goal_state == 7 * 7 - 1


def test_maze_bfs_positive():
    env = Maze(width=7, height=7, seed=0)
    path_len = env.bfs_shortest()
    assert path_len > 0


def test_maze_step_done_at_goal():
    env = Maze(width=5, height=5, seed=42)
    env.reset()
    env.agent_state = env.goal_state
    # Any action from goal triggers done
    _, reward, done = env.step(0)
    assert done is True
    assert reward == 10.0


def test_maze_n_states():
    env = Maze(width=9, height=9, seed=1)
    assert env.n_states == 81
```

- [ ] **Step 2: Run Maze tests — expect FAIL**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/test_environments.py -k "maze" -v 2>&1 | head -20
```

- [ ] **Step 3: Implement maze.py**

`core/environments/maze.py`:
```python
"""
Maze — Recursive Backtracking perfect maze + BFS solver. Zero Qt imports.
SRS-RLAI-001 FR-PG-MZ-01
"""
import numpy as np
from collections import deque


class Maze:
    # Actions: 0=Up, 1=Down, 2=Left, 3=Right
    # Directions: (dy, dx) in grid coordinates (y increases downward)
    DIRS = [(0, -1, 0), (0, 1, 1), (-1, 0, 2), (1, 0, 3)]  # (dy, dx, action_idx)
    # Wall indices per cell: [North, South, West, East] → maps to action complement
    WALL_N, WALL_S, WALL_W, WALL_E = 0, 1, 2, 3
    ACTION_TO_WALL = {0: WALL_N, 1: WALL_S, 2: WALL_W, 3: WALL_E}
    OPP_WALL = {WALL_N: WALL_S, WALL_S: WALL_N, WALL_W: WALL_E, WALL_E: WALL_W}

    def __init__(self, width: int = 15, height: int = 15, seed: int = None):
        self.width  = width
        self.height = height
        self.n_states  = width * height
        self.n_actions = 4
        self.start_state = 0
        self.goal_state  = self.n_states - 1
        self.agent_state = 0
        self.rng = np.random.default_rng(seed)
        # walls[y, x, direction] = True means wall exists (blocked)
        self.walls = np.ones((height, width, 4), dtype=bool)
        self._generate()

    def _generate(self):
        """Recursive Backtracking (iterative DFS)."""
        visited = np.zeros((self.height, self.width), dtype=bool)
        stack = [(0, 0)]
        visited[0, 0] = True
        while stack:
            y, x = stack[-1]
            dirs = [(dy, dx, a) for dy, dx, a in self.DIRS]
            self.rng.shuffle(dirs)
            moved = False
            for dy, dx, _ in dirs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width and not visited[ny, nx]:
                    # Remove wall between (y,x) and (ny,nx)
                    w_from = self._dir_to_wall(dy, dx)
                    w_to   = self.OPP_WALL[w_from]
                    self.walls[y,  x,  w_from] = False
                    self.walls[ny, nx, w_to]   = False
                    visited[ny, nx] = True
                    stack.append((ny, nx))
                    moved = True
                    break
            if not moved:
                stack.pop()

    def _dir_to_wall(self, dy: int, dx: int) -> int:
        if dy == -1: return self.WALL_N
        if dy ==  1: return self.WALL_S
        if dx == -1: return self.WALL_W
        return self.WALL_E

    def _state_to_yx(self, state: int) -> tuple:
        return divmod(state, self.width)

    def _yx_to_state(self, y: int, x: int) -> int:
        return y * self.width + x

    def reset(self) -> int:
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action: int) -> tuple:
        if self.agent_state == self.goal_state:
            return self.agent_state, 10.0, True
        y, x = self._state_to_yx(self.agent_state)
        wall_idx = self.ACTION_TO_WALL[action]
        if self.walls[y, x, wall_idx]:
            return self.agent_state, -1.0, False   # hit wall
        dy_map = {self.WALL_N: -1, self.WALL_S: 1, self.WALL_W: 0,  self.WALL_E: 0}
        dx_map = {self.WALL_N: 0,  self.WALL_S: 0, self.WALL_W: -1, self.WALL_E: 1}
        ny = y + dy_map[wall_idx]; nx = x + dx_map[wall_idx]
        self.agent_state = self._yx_to_state(ny, nx)
        if self.agent_state == self.goal_state:
            return self.agent_state, 10.0, True
        return self.agent_state, -0.1, False

    def bfs_shortest(self) -> int:
        """Returns number of steps in BFS shortest path from start to goal."""
        visited = {self.start_state}
        queue   = deque([(self.start_state, 0)])
        while queue:
            state, dist = queue.popleft()
            if state == self.goal_state: return dist
            y, x = self._state_to_yx(state)
            for action in range(4):
                wall_idx = self.ACTION_TO_WALL[action]
                if not self.walls[y, x, wall_idx]:
                    dy_m = {self.WALL_N:-1,self.WALL_S:1,self.WALL_W:0,self.WALL_E:0}
                    dx_m = {self.WALL_N:0,self.WALL_S:0,self.WALL_W:-1,self.WALL_E:1}
                    ns = self._yx_to_state(y+dy_m[wall_idx], x+dx_m[wall_idx])
                    if ns not in visited:
                        visited.add(ns); queue.append((ns, dist+1))
        return -1  # unreachable (shouldn't happen in perfect maze)

    def get_walls(self) -> np.ndarray:
        return self.walls.copy()
```

- [ ] **Step 4: Run all environment tests — expect PASS**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/test_environments.py -v
```

- [ ] **Step 5: Commit**
```bash
git add core/environments/maze.py tests/test_environments.py
git commit -m "feat: Maze environment — Recursive Backtracking + BFS (FR-PG-MZ-01)"
```

---

## Task 5: Q-Learning Algorithm + Tests

**Files:**
- Create: `core/algorithms/qlearning.py`
- Create: `tests/test_algorithms.py`

- [ ] **Step 1: Write failing tests**

`tests/test_algorithms.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from core.environments.gridworld import GridWorld
from core.algorithms.qlearning import QLearning


def test_qlearning_qtable_init_zeros():
    ql = QLearning(n_states=4, n_actions=4)
    assert ql.q_table.shape == (4, 4)
    assert np.all(ql.q_table == 0.0)


def test_qlearning_select_action_random_when_epsilon_1():
    ql = QLearning(n_states=4, n_actions=4, epsilon=1.0)
    np.random.seed(0)
    actions = [ql.select_action(0) for _ in range(100)]
    assert len(set(actions)) > 1  # not all same


def test_qlearning_select_action_greedy_when_epsilon_0():
    ql = QLearning(n_states=4, n_actions=4, epsilon=0.0)
    ql.q_table[0] = [0.0, 0.0, 0.0, 5.0]  # best action = 3
    action = ql.select_action(0)
    assert action == 3


def test_qlearning_update_increases_q():
    ql = QLearning(n_states=4, n_actions=4, alpha=1.0, gamma=0.9, epsilon=0.0)
    ql.update(0, 3, 1.0, 1, False)  # Q[0,3] ← 0 + 1.0*(1+0*0-0) = 1.0
    assert abs(ql.q_table[0, 3] - 1.0) < 1e-6


def test_qlearning_run_episode_returns_tuple():
    env = GridWorld(n=4)
    ql  = QLearning(n_states=env.n_states, n_actions=env.n_actions, epsilon=1.0)
    reward, steps, traj = ql.run_episode(env)
    assert isinstance(reward, float)
    assert isinstance(steps, int)
    assert len(traj) > 0


def test_qlearning_converges_on_small_gridworld():
    """After 2000 episodes on 2×2, agent should reliably reach goal."""
    env = GridWorld(n=2)
    ql  = QLearning(n_states=env.n_states, n_actions=env.n_actions,
                    alpha=0.5, gamma=0.9, epsilon=1.0,
                    epsilon_decay=0.99, epsilon_min=0.01)
    for _ in range(2000):
        ql.run_episode(env)
    # After training, greedy policy from state 0 should reach goal
    ql.epsilon = 0.0
    env.reset()
    total_r, steps, _ = ql.run_episode(env)
    assert total_r > 0  # reached goal


def test_qlearning_get_policy_shape():
    ql = QLearning(n_states=36, n_actions=4)
    p  = ql.get_policy()
    assert p.shape == (36,)


def test_qlearning_epsilon_decays():
    ql = QLearning(n_states=4, n_actions=4, epsilon=1.0,
                   epsilon_decay=0.9, epsilon_min=0.01)
    env = GridWorld(n=2)
    ql.run_episode(env)
    assert ql.epsilon < 1.0
```

- [ ] **Step 2: Run — expect FAIL**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/test_algorithms.py -v 2>&1 | head -20
```

- [ ] **Step 3: Implement qlearning.py**

`core/algorithms/qlearning.py`:
```python
"""
QLearning — Tabular Q-Learning with ε-greedy exploration. Zero Qt imports.
SRS-RLAI-001 FR-CONC-VB-01..04, FR-PG-GW-09
"""
import numpy as np


class QLearning:
    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        self.n_states     = n_states
        self.n_actions    = n_actions
        self.alpha        = alpha
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min  = epsilon_min
        self.q_table      = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool):
        target = reward + self.gamma * np.max(self.q_table[next_state]) * (1 - done)
        td_err = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_err
        return float(td_err)

    def run_episode(self, env, max_steps: int = 200) -> tuple:
        """Returns (total_reward, steps, trajectory).
        trajectory: list of (state, action, reward).
        """
        state = env.reset()
        total_reward = 0.0
        trajectory   = []
        for step in range(max_steps):
            action = self.select_action(state)
            next_state, reward, done = env.step(action)
            self.update(state, action, reward, next_state, done)
            trajectory.append((state, action, reward))
            total_reward += reward
            state = next_state
            if done:
                break
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return total_reward, step + 1, trajectory

    def get_policy(self) -> np.ndarray:
        """Returns argmax action per state. Shape (n_states,)."""
        return np.argmax(self.q_table, axis=1)

    def get_q_values(self) -> np.ndarray:
        return self.q_table.copy()

    def reset(self):
        self.q_table[:] = 0.0
        self.epsilon = 1.0
```

- [ ] **Step 4: Run — expect PASS**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/test_algorithms.py -v
```

- [ ] **Step 5: Commit**
```bash
git add core/algorithms/qlearning.py tests/test_algorithms.py
git commit -m "feat: QLearning tabular algorithm (FR-CONC-VB-01..04)"
```

---

## Task 6: REINFORCE Algorithm + Tests

**Files:**
- Create: `core/algorithms/reinforce.py`
- Modify: `tests/test_algorithms.py` (append REINFORCE tests)

- [ ] **Step 1: Append REINFORCE tests**

Append to `tests/test_algorithms.py`:
```python
from core.algorithms.reinforce import REINFORCE, LinearPolicy


def test_linear_policy_output_sums_to_1():
    pol = LinearPolicy(n_states=4, n_actions=4)
    probs = pol.forward(0)
    assert abs(probs.sum() - 1.0) < 1e-6


def test_linear_policy_output_shape():
    pol = LinearPolicy(n_states=36, n_actions=4)
    probs = pol.forward(5)
    assert probs.shape == (4,)


def test_reinforce_select_action_valid():
    rf = REINFORCE(n_states=4, n_actions=4)
    action, log_prob = rf.select_action(0)
    assert action in range(4)
    assert log_prob < 0  # log(prob) < 0 for prob < 1


def test_reinforce_compute_returns_discounted():
    rf = REINFORCE(n_states=4, n_actions=4, gamma=0.5)
    # rewards = [1, 1, 1]; G[0]=1+0.5+0.25=1.75, G[1]=1+0.5=1.5, G[2]=1
    G = rf.compute_returns([1.0, 1.0, 1.0])
    assert abs(G[0] - 1.75) < 1e-6
    assert abs(G[1] - 1.5) < 1e-6
    assert abs(G[2] - 1.0) < 1e-6


def test_reinforce_run_episode_returns_tuple():
    env = GridWorld(n=4)
    rf  = REINFORCE(n_states=env.n_states, n_actions=env.n_actions)
    reward, steps, traj = rf.run_episode(env)
    assert isinstance(reward, float)
    assert steps > 0
    assert len(traj) > 0


def test_reinforce_get_policy_probs_shape():
    rf = REINFORCE(n_states=36, n_actions=4)
    probs = rf.get_policy_probs()
    assert probs.shape == (36, 4)
    assert np.allclose(probs.sum(axis=1), 1.0)
```

- [ ] **Step 2: Run REINFORCE tests — expect FAIL**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/test_algorithms.py -k "reinforce or linear_policy" -v 2>&1 | head -20
```

- [ ] **Step 3: Implement reinforce.py**

`core/algorithms/reinforce.py`:
```python
"""
REINFORCE — Monte Carlo Policy Gradient with linear softmax policy. Zero Qt imports.
SRS-RLAI-001 FR-CONC-PB-01..04, FR-PG-GW-09
"""
import numpy as np


class LinearPolicy:
    """Softmax linear policy: π(a|s) = softmax(W·one_hot(s) + b)"""

    def __init__(self, n_states: int, n_actions: int):
        self.n_states  = n_states
        self.n_actions = n_actions
        # Small random init
        rng = np.random.default_rng(42)
        self.W = rng.normal(0, 0.01, (n_actions, n_states))
        self.b = np.zeros(n_actions)

    def _one_hot(self, state: int) -> np.ndarray:
        oh = np.zeros(self.n_states)
        oh[state] = 1.0
        return oh

    def forward(self, state: int) -> np.ndarray:
        """Returns probability distribution over actions. Shape (n_actions,)."""
        logits = self.W @ self._one_hot(state) + self.b
        return self._softmax(logits)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        ex = np.exp(x - np.max(x))  # numerical stability
        return ex / ex.sum()

    def log_prob(self, state: int, action: int) -> float:
        probs = self.forward(state)
        return float(np.log(probs[action] + 1e-10))

    def gradient(self, state: int, action: int) -> tuple:
        """Returns (dW, db) for log π(action|state). Shape matches W, b."""
        probs   = self.forward(state)
        one_hot_s = self._one_hot(state)
        one_hot_a = np.zeros(self.n_actions); one_hot_a[action] = 1.0
        # ∇log π = one_hot(a) - π  (softmax cross-entropy gradient)
        d_logits = one_hot_a - probs
        dW = np.outer(d_logits, one_hot_s)
        db = d_logits.copy()
        return dW, db

    def reset(self):
        rng = np.random.default_rng(42)
        self.W = rng.normal(0, 0.01, (self.n_actions, self.n_states))
        self.b = np.zeros(self.n_actions)


class REINFORCE:
    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.001, gamma: float = 0.99):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.policy    = LinearPolicy(n_states, n_actions)

    def select_action(self, state: int) -> tuple:
        """Returns (action, log_prob)."""
        probs  = self.policy.forward(state)
        action = int(np.random.choice(self.n_actions, p=probs))
        return action, self.policy.log_prob(state, action)

    def compute_returns(self, rewards: list) -> np.ndarray:
        """G[t] = Σ_{k=t}^{T} γ^(k-t) r[k]  (backward pass)."""
        T = len(rewards)
        G = np.zeros(T)
        G[-1] = rewards[-1]
        for t in range(T - 2, -1, -1):
            G[t] = rewards[t] + self.gamma * G[t + 1]
        return G

    def update(self, states: list, actions: list, returns: np.ndarray):
        """Normalize returns and apply policy gradient update."""
        G = (returns - returns.mean()) / (returns.std() + 1e-8)
        dW_total = np.zeros_like(self.policy.W)
        db_total = np.zeros_like(self.policy.b)
        for state, action, g in zip(states, actions, G):
            dW, db = self.policy.gradient(state, action)
            dW_total += g * dW
            db_total += g * db
        self.policy.W += self.alpha * dW_total
        self.policy.b += self.alpha * db_total

    def run_episode(self, env, max_steps: int = 200) -> tuple:
        """Returns (total_reward, steps, trajectory).
        trajectory: list of (state, action, reward).
        """
        state   = env.reset()
        states, actions, rewards, trajectory = [], [], [], []
        for step in range(max_steps):
            action, log_prob = self.select_action(state)
            next_state, reward, done = env.step(action)
            states.append(state); actions.append(action); rewards.append(reward)
            trajectory.append((state, action, reward))
            state = next_state
            if done: break
        returns = self.compute_returns(rewards)
        self.update(states, actions, returns)
        return sum(rewards), step + 1, trajectory

    def get_policy_probs(self) -> np.ndarray:
        """Returns π(a|s) for all states. Shape (n_states, n_actions)."""
        return np.array([self.policy.forward(s) for s in range(self.n_states)])

    def reset(self):
        self.policy.reset()
```

- [ ] **Step 4: Run all algorithm tests — expect PASS**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/test_algorithms.py -v
```

- [ ] **Step 5: Commit**
```bash
git add core/algorithms/reinforce.py tests/test_algorithms.py
git commit -m "feat: REINFORCE + LinearPolicy (FR-CONC-PB-01..04)"
```

---

## Task 7: ReplayBuffer + Tests

**Files:**
- Create: `core/algorithms/replay_buffer.py`
- Create: `tests/test_replay_buffer.py`

- [ ] **Step 1: Write failing tests**

`tests/test_replay_buffer.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from core.algorithms.replay_buffer import ReplayBuffer


def test_buffer_empty_on_init():
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0


def test_buffer_push_increases_len():
    buf = ReplayBuffer(capacity=10)
    buf.push(0, 1, 1.0, 1, False)
    assert len(buf) == 1


def test_buffer_circular_overflow():
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.push(i, 0, float(i), i+1, False)
    assert len(buf) == 3   # capped at capacity


def test_buffer_sample_shape():
    buf = ReplayBuffer(capacity=100)
    for i in range(20):
        buf.push(i, i % 4, float(i), (i+1) % 20, i == 19)
    s, a, r, ns, d = buf.sample(8)
    assert len(s) == 8 and len(a) == 8 and len(r) == 8


def test_buffer_sample_requires_enough_data():
    buf = ReplayBuffer(capacity=10)
    buf.push(0, 0, 1.0, 1, False)
    with pytest.raises(ValueError):
        buf.sample(5)


def test_buffer_ptr_wraps():
    buf = ReplayBuffer(capacity=3)
    for i in range(6):
        buf.push(i, 0, float(i), i+1, False)
    assert buf.ptr == 0  # ptr wrapped twice around capacity=3
```

- [ ] **Step 2: Run — expect FAIL**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/test_replay_buffer.py -v 2>&1 | head -20
```

- [ ] **Step 3: Implement replay_buffer.py**

`core/algorithms/replay_buffer.py`:
```python
"""
ReplayBuffer — Fixed-size circular experience replay buffer for DQN concept visualization.
SRS-RLAI-001 FR-CONC-VB-06
"""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.ptr      = 0
        self.size     = 0
        self._states      = np.zeros(capacity, dtype=np.int32)
        self._actions     = np.zeros(capacity, dtype=np.int32)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._next_states = np.zeros(capacity, dtype=np.int32)
        self._dones       = np.zeros(capacity, dtype=bool)

    def push(self, state: int, action: int, reward: float,
             next_state: int, done: bool):
        self._states[self.ptr]      = state
        self._actions[self.ptr]     = action
        self._rewards[self.ptr]     = reward
        self._next_states[self.ptr] = next_state
        self._dones[self.ptr]       = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        if self.size < batch_size:
            raise ValueError(f"Not enough data: {self.size} < {batch_size}")
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (self._states[idx], self._actions[idx], self._rewards[idx],
                self._next_states[idx], self._dones[idx])

    def __len__(self) -> int:
        return self.size

    def get_all_valid(self) -> tuple:
        """Returns all stored transitions (for visualization)."""
        idx = np.arange(self.size)
        return (self._states[idx], self._actions[idx], self._rewards[idx],
                self._next_states[idx], self._dones[idx])
```

- [ ] **Step 4: Run — expect PASS**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/ -v
```

- [ ] **Step 5: Commit**
```bash
git add core/algorithms/replay_buffer.py tests/test_replay_buffer.py
git commit -m "feat: ReplayBuffer circular buffer (FR-CONC-VB-06)"
```

---

## Task 8: Theme + PainterUtils

**Files:**
- Create: `ui/theme.py`
- Create: `ui/visualizations/painter_utils.py`

- [ ] **Step 1: Implement theme.py**

`ui/theme.py`:
```python
"""
Theme — Glassmorphism dark color tokens + QSS. Single source of truth.
SRS-RLAI-001 FR-UI-02, NFR-VQ-01..04
"""
import math
from PySide6.QtGui import QColor, QRadialGradient, QLinearGradient
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

# ── Color Tokens ──────────────────────────────────────────────────────────────
BG        = QColor(10,  14,  26)       # #0A0E1A
SURFACE1  = QColor(22,  28,  45)       # #161C2D
SURFACE2  = QColor(30,  38,  58)       # #1E263A
BORDER    = QColor(255, 255, 255, 40)
BORDER_HI = QColor(255, 255, 255, 80)

CYAN      = QColor(0,   212, 255)      # Q-Learning / primary
MAGENTA   = QColor(255, 0,   110)      # REINFORCE / danger
VIOLET    = QColor(124, 58,  237)      # Policy / DQN
EMERALD   = QColor(16,  185, 129)      # Reward / Goal
AMBER     = QColor(245, 158, 11)       # Epsilon / warning
WHITE_80  = QColor(255, 255, 255, 204)
WHITE_60  = QColor(255, 255, 255, 153)
WHITE_40  = QColor(255, 255, 255, 102)
WHITE_20  = QColor(255, 255, 255, 51)

HEATMAP_LOW  = QColor(255, 0,   110, 60)
HEATMAP_HIGH = QColor(0,   212, 255, 60)

# ── QSS ───────────────────────────────────────────────────────────────────────
QSS = """
QMainWindow, QDialog { background: #0A0E1A; }
QWidget { background: transparent; color: #E2E8F0; font-size: 11pt; font-family: 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif; }
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: #161C2D; width: 6px; border-radius: 3px; margin: 0; }
QScrollBar::handle:vertical { background: #2D3748; border-radius: 3px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QPushButton {
    background: rgba(0,212,255,0.10); color: #00D4FF;
    border: 1px solid rgba(0,212,255,0.25); border-radius: 8px;
    padding: 8px 20px; font-weight: 700; font-size: 10pt;
}
QPushButton:hover { background: rgba(0,212,255,0.22); border-color: rgba(0,212,255,0.5); }
QPushButton:pressed { background: rgba(0,212,255,0.35); }
QPushButton:disabled { background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.3); border-color: rgba(255,255,255,0.1); }
QPushButton#danger { background: rgba(255,0,110,0.10); color: #FF006E; border-color: rgba(255,0,110,0.25); }
QPushButton#danger:hover { background: rgba(255,0,110,0.22); }
QPushButton#success { background: rgba(16,185,129,0.10); color: #10B981; border-color: rgba(16,185,129,0.25); }
QSlider::groove:horizontal { background: #1E263A; height: 6px; border-radius: 3px; }
QSlider::sub-page:horizontal { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #7C3AED,stop:1 #00D4FF); border-radius: 3px; }
QSlider::handle:horizontal { background: #00D4FF; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; border: 2px solid #0A0E1A; }
QLabel { background: transparent; }
QComboBox { background: #161C2D; border: 1px solid rgba(255,255,255,0.15); border-radius: 6px; padding: 4px 10px; color: #E2E8F0; }
QComboBox::drop-down { border: none; }
QTabBar::tab { background: transparent; color: rgba(255,255,255,0.5); padding: 10px 20px; font-size: 10pt; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #00D4FF; border-bottom: 2px solid #00D4FF; font-weight: 700; }
QTabWidget::pane { border: none; }
"""


def apply(app: QApplication):
    app.setStyleSheet(QSS)


def lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        int(c1.red()   + (c2.red()   - c1.red())   * t),
        int(c1.green() + (c2.green() - c1.green()) * t),
        int(c1.blue()  + (c2.blue()  - c1.blue())  * t),
        int(c1.alpha() + (c2.alpha() - c1.alpha()) * t),
    )


def ease_in_out(t: float) -> float:
    """Quadratic ease-in-out. t ∈ [0, 1]."""
    t = max(0.0, min(1.0, t))
    return 2*t*t if t < 0.5 else -1 + (4 - 2*t)*t


def pulse(phase: float, lo: float = 0.5, hi: float = 1.0) -> float:
    """Smooth pulse between lo and hi at given phase [0,1)."""
    return lo + (hi - lo) * (0.5 + 0.5 * math.cos(2 * math.pi * phase))
```

- [ ] **Step 2: Implement painter_utils.py**

`ui/visualizations/painter_utils.py`:
```python
"""
PainterUtils — shared QPainter drawing helpers. No business logic.
SRS-RLAI-001 NFR-VQ-03, NFR-VQ-04
"""
import math
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QRadialGradient, QFont, QFontMetrics


def draw_glow(p: QPainter, center: QPointF, radius: float, color: QColor):
    """3-layer neon glow: outer, inner, core. SDD-RLAI-001 §7.3"""
    # Outer glow
    g1 = QRadialGradient(center, radius * 2.5)
    g1.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 80))
    g1.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setBrush(g1); p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(center, radius * 2.5, radius * 2.5)
    # Inner glow
    g2 = QRadialGradient(center, radius * 1.5)
    g2.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 160))
    g2.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setBrush(g2)
    p.drawEllipse(center, radius * 1.5, radius * 1.5)
    # Core
    p.setBrush(QBrush(color))
    p.drawEllipse(center, radius, radius)


def draw_arrow(p: QPainter, cx: float, cy: float, direction: int,
               size: float, color: QColor, opacity: float = 1.0):
    """Draw directional arrow. direction: 0=Up,1=Down,2=Left,3=Right."""
    old_opacity = p.opacity()
    p.setOpacity(opacity * old_opacity)
    p.setPen(QPen(color, 2.0)); p.setBrush(QBrush(color))
    # Arrowhead as triangle; direction determines rotation
    angle_map = {0: -90, 1: 90, 2: 180, 3: 0}
    angle = math.radians(angle_map[direction])
    tip_x = cx + size * 0.5 * math.cos(angle)
    tip_y = cy + size * 0.5 * math.sin(angle)
    tail_x = cx - size * 0.35 * math.cos(angle)
    tail_y = cy - size * 0.35 * math.sin(angle)
    perp = angle + math.pi / 2
    w = size * 0.25
    from PySide6.QtGui import QPolygonF
    poly = QPolygonF([
        QPointF(tip_x, tip_y),
        QPointF(tail_x + w * math.cos(perp), tail_y + w * math.sin(perp)),
        QPointF(tail_x - w * math.cos(perp), tail_y - w * math.sin(perp)),
    ])
    p.drawPolygon(poly)
    p.setOpacity(old_opacity)


def draw_glass_rect(p: QPainter, rect: QRectF, radius: float = 12.0):
    """Glassmorphism panel background."""
    p.setBrush(QBrush(QColor(255, 255, 255, 15)))
    p.setPen(QPen(QColor(255, 255, 255, 40), 1.0))
    p.drawRoundedRect(rect, radius, radius)


def draw_text_center(p: QPainter, rect: QRectF, text: str, color: QColor,
                     font_size: int = 11, bold: bool = False):
    f = QFont(); f.setPointSize(font_size); f.setBold(bold)
    p.setFont(f); p.setPen(color)
    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


def heatmap_color(value: float, min_val: float, max_val: float) -> QColor:
    """Maps value to cyan-magenta heatmap color."""
    from ui.theme import HEATMAP_LOW, HEATMAP_HIGH, lerp_color
    if max_val == min_val: return QColor(30, 38, 58, 200)
    t = (value - min_val) / (max_val - min_val)
    return lerp_color(HEATMAP_LOW, HEATMAP_HIGH, t)
```

- [ ] **Step 3: Smoke test imports**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -c "
import sys; sys.path.insert(0,'.')
import matplotlib; matplotlib.use('QtAgg') if False else None
from PySide6.QtWidgets import QApplication; app = QApplication([])
from ui.theme import apply, CYAN, MAGENTA
from ui.visualizations.painter_utils import draw_glow, draw_arrow
print('theme + painter_utils OK')
"
```

- [ ] **Step 4: Commit**
```bash
git add ui/theme.py ui/visualizations/painter_utils.py
git commit -m "feat: Glassmorphism theme tokens + QPainter utilities (NFR-VQ-01..04)"
```

---

## Task 9: UI Components

**Files:**
- Create: `ui/components/glass_panel.py`
- Create: `ui/components/slider_group.py`
- Create: `ui/components/learning_curve.py`
- Create: `ui/components/status_bar.py`

- [ ] **Step 1: glass_panel.py**

`ui/components/glass_panel.py`:
```python
"""GlassPanel — Glassmorphism base widget. Draws frosted background in paintEvent."""
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtGui import QPainter, QPen, QBrush, QColor
from PySide6.QtCore import QRectF


class GlassPanel(QWidget):
    def __init__(self, parent=None, radius: float = 12.0, alpha: int = 15):
        super().__init__(parent)
        self._radius = radius
        self._alpha  = alpha

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        p.setBrush(QBrush(QColor(255, 255, 255, self._alpha)))
        p.setPen(QPen(QColor(255, 255, 255, 40), 1.0))
        p.drawRoundedRect(rect, self._radius, self._radius)
        p.end()
        super().paintEvent(event)
```

- [ ] **Step 2: slider_group.py**

`ui/components/slider_group.py`:
```python
"""SliderGroup — labeled QSlider with value display, neon-styled."""
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                QLabel, QSlider, QDoubleSpinBox)
from PySide6.QtGui import QFont
from ui.components.glass_panel import GlassPanel


class SliderGroup(GlassPanel):
    value_changed = Signal(str, float)   # (param_name, value)

    def __init__(self, name: str, label: str, min_val: float, max_val: float,
                 default: float, step: float = 0.01, decimals: int = 3,
                 unit: str = "", parent=None):
        super().__init__(parent, radius=8, alpha=10)
        self._name    = name
        self._default = default
        self._blocking = False

        lay = QVBoxLayout(self); lay.setContentsMargins(12, 10, 12, 10); lay.setSpacing(4)

        # Label row
        lbl_row = QHBoxLayout()
        lbl = QLabel(label + (f" [{unit}]" if unit else ""))
        f = QFont(); f.setWeight(QFont.Weight.Medium); lbl.setFont(f)
        lbl.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 9pt;")
        self._val_lbl = QLabel(f"{default:.{decimals}f}")
        self._val_lbl.setStyleSheet("color: #00D4FF; font-weight: 700; font-size: 9pt;")
        lbl_row.addWidget(lbl); lbl_row.addStretch(); lbl_row.addWidget(self._val_lbl)
        lay.addLayout(lbl_row)

        # Slider
        n_steps = max(1, round((max_val - min_val) / step))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, n_steps)
        self._slider.setValue(round((default - min_val) / step))
        lay.addWidget(self._slider)

        self._min = min_val; self._step = step; self._decimals = decimals
        self._slider.valueChanged.connect(self._on_slider)

    def _on_slider(self, i: int):
        if self._blocking: return
        v = self._min + i * self._step
        self._val_lbl.setText(f"{v:.{self._decimals}f}")
        self.value_changed.emit(self._name, v)

    @property
    def value(self) -> float:
        return self._min + self._slider.value() * self._step

    @value.setter
    def value(self, v: float):
        self._blocking = True
        self._slider.setValue(round((v - self._min) / self._step))
        self._val_lbl.setText(f"{v:.{self._decimals}f}")
        self._blocking = False
```

- [ ] **Step 3: learning_curve.py**

`ui/components/learning_curve.py`:
```python
"""LearningCurveWidget — QPainter line chart with moving average."""
import math
from collections import deque
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QPen, QColor, QPainterPath, QFont
from PySide6.QtCore import Qt, QRectF
from ui.theme import CYAN, MAGENTA, WHITE_20, WHITE_40


class LearningCurveWidget(QWidget):
    def __init__(self, max_points: int = 500, color: QColor = None,
                 label: str = "Episode Reward", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self._rewards  = deque(maxlen=max_points)
        self._ma       = deque(maxlen=max_points)   # 50-ep moving average
        self._color    = color or CYAN
        self._label    = label
        self._ma_win   = 50

    def add_episode(self, reward: float):
        self._rewards.append(reward)
        window = list(self._rewards)[-self._ma_win:]
        self._ma.append(sum(window) / len(window))
        self.update()

    def clear(self):
        self._rewards.clear(); self._ma.clear(); self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        PAD = 36

        # Background
        p.fillRect(self.rect(), QColor(10, 14, 26))

        if len(self._rewards) < 2:
            p.setPen(WHITE_40); f = QFont(); f.setPointSize(9); p.setFont(f)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Training data will appear here…")
            p.end(); return

        rewards = list(self._rewards)
        min_r = min(rewards); max_r = max(rewards)
        span  = max(max_r - min_r, 1e-6)
        n     = len(rewards)

        def to_xy(i, r):
            x = PAD + (i / (n - 1)) * (W - PAD - 8)
            y = (H - PAD) - ((r - min_r) / span) * (H - PAD - 16)
            return x, y

        # Grid lines
        p.setPen(QPen(WHITE_20, 0.5, Qt.PenStyle.DashLine))
        for frac in [0.25, 0.5, 0.75, 1.0]:
            y = (H - PAD) - frac * (H - PAD - 16)
            p.drawLine(PAD, int(y), W - 8, int(y))
            r_val = min_r + frac * span
            f = QFont(); f.setPointSize(8); p.setFont(f)
            p.setPen(WHITE_40)
            p.drawText(2, int(y) + 4, f"{r_val:.1f}")
            p.setPen(QPen(WHITE_20, 0.5, Qt.PenStyle.DashLine))

        # Axes
        p.setPen(QPen(WHITE_40, 1.0))
        p.drawLine(PAD, 8, PAD, H - PAD)
        p.drawLine(PAD, H - PAD, W - 8, H - PAD)

        # Reward series (semi-transparent)
        c_raw = QColor(self._color); c_raw.setAlpha(80)
        path = QPainterPath()
        for i, r in enumerate(rewards):
            x, y = to_xy(i, r)
            if i == 0: path.moveTo(x, y)
            else: path.lineTo(x, y)
        p.setPen(QPen(c_raw, 1.2)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

        # Moving average (solid)
        ma = list(self._ma)
        if len(ma) >= 2:
            ma_path = QPainterPath()
            for i, r in enumerate(ma):
                x, y = to_xy(i, r)
                if i == 0: ma_path.moveTo(x, y)
                else: ma_path.lineTo(x, y)
            p.setPen(QPen(self._color, 2.2))
            p.drawPath(ma_path)

        # Label
        f = QFont(); f.setPointSize(8); f.setBold(True); p.setFont(f)
        p.setPen(self._color)
        p.drawText(PAD + 4, 14, f"{self._label}  (MA50: {ma[-1]:.2f})" if ma else self._label)
        p.end()
```

- [ ] **Step 4: status_bar.py**

`ui/components/status_bar.py`:
```python
"""StatusBar — episode counter, algorithm status, speed indicator."""
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
from ui.theme import CYAN, AMBER


class StatusBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        lay = QHBoxLayout(self); lay.setContentsMargins(16, 0, 16, 0)
        self._ep_lbl    = QLabel("Episode: —")
        self._reward_lbl = QLabel("Reward: —")
        self._eps_lbl   = QLabel("ε: —")
        self._speed_lbl = QLabel("⚡ Normal")
        for lbl in (self._ep_lbl, self._reward_lbl, self._eps_lbl):
            lbl.setStyleSheet("color: rgba(255,255,255,0.6); font-size: 9pt;")
        self._speed_lbl.setStyleSheet("color: #00D4FF; font-size: 9pt; font-weight: 700;")
        lay.addWidget(self._ep_lbl); lay.addWidget(self._reward_lbl)
        lay.addWidget(self._eps_lbl); lay.addStretch(); lay.addWidget(self._speed_lbl)

    def update_step(self, data: dict):
        ep = data.get('episode', '—')
        r  = data.get('total_reward', 0)
        eps = data.get('epsilon', 0)
        self._ep_lbl.setText(f"Episode: {ep}")
        self._reward_lbl.setText(f"Reward: {r:.2f}")
        self._eps_lbl.setText(f"ε: {eps:.3f}")

    def set_speed(self, label: str):
        self._speed_lbl.setText(f"⚡ {label}")
```

- [ ] **Step 5: Commit**
```bash
git add ui/components/
git commit -m "feat: UI components — GlassPanel, SliderGroup, LearningCurve, StatusBar"
```

---

## Task 10: Sidebar

**Files:**
- Create: `ui/components/sidebar.py`

- [ ] **Step 1: Implement sidebar.py**

`ui/components/sidebar.py`:
```python
"""Sidebar — Glassmorphism navigation sidebar with expandable groups."""
import math
from PySide6.QtCore import Signal, Qt, QRectF, QPointF
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QPushButton, QLabel
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QLinearGradient
from ui.theme import BG, SURFACE1, SURFACE2, CYAN, MAGENTA, VIOLET, WHITE_40, WHITE_60, pulse


NAV_ITEMS = [
    ("🧠  Concepts", None, [
        ("RL Basics",    "concepts/rl-basics"),
        ("Value-Based",  "concepts/value-based"),
        ("Policy-Based", "concepts/policy-based"),
        ("Applications", "concepts/applications"),
    ]),
    ("🎮  Playground", None, [
        ("GridWorld", "playground/gridworld"),
        ("CartPole",  "playground/cartpole"),
        ("Maze",      "playground/maze"),
    ]),
    ("⚔️  Comparison Arena", "arena", []),
]


class SidebarItem(QWidget):
    clicked = Signal(str)

    def __init__(self, label: str, page_id: str, indent: int = 0, parent=None):
        super().__init__(parent)
        self.label   = label
        self.page_id = page_id
        self.indent  = indent
        self._active = False
        self._hover  = False
        self._phase  = 0.0
        self.setFixedHeight(36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_active(self, v: bool): self._active = v; self.update()
    def set_phase(self, v: float): self._phase = v; self.update()

    def enterEvent(self, e): self._hover = True;  self.update()
    def leaveEvent(self, e): self._hover = False; self.update()
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self.clicked.emit(self.page_id)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        if self._active:
            # Glow background
            g = QLinearGradient(0, 0, W, 0)
            g.setColorAt(0, QColor(0, 212, 255, 30))
            g.setColorAt(1, QColor(0, 212, 255, 5))
            p.fillRect(self.rect(), g)
            # Left accent bar
            br = pulse(self._phase, 0.7, 1.0)
            p.fillRect(0, 4, 3, H - 8, QColor(0, int(212*br), int(255*br)))
        elif self._hover:
            p.fillRect(self.rect(), QColor(255, 255, 255, 8))

        # Text
        color = CYAN if self._active else (WHITE_60 if self._hover else WHITE_40)
        f = QFont(); f.setPointSize(10)
        if self._active: f.setBold(True)
        p.setFont(f); p.setPen(color)
        p.drawText(QRectF(12 + self.indent * 16, 0, W - 24, H),
                   Qt.AlignmentFlag.AlignVCenter, self.label)
        p.end()


class Sidebar(QWidget):
    page_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        self._items: list[SidebarItem] = []
        self._active_id = "concepts/rl-basics"
        self._phase = 0.0
        self._build()

    def _build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)

        # Logo area
        logo = QLabel("RL Dashboard")
        logo.setFixedHeight(60)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet("color: #00D4FF; font-size: 13pt; font-weight: 900; "
                           "border-bottom: 1px solid rgba(255,255,255,0.08);")
        lay.addWidget(logo)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(); inner_lay = QVBoxLayout(inner)
        inner_lay.setContentsMargins(0, 8, 0, 8); inner_lay.setSpacing(0)

        for group_label, group_id, children in NAV_ITEMS:
            # Group header
            grp = QLabel(f"  {group_label}")
            grp.setFixedHeight(40)
            grp.setStyleSheet("color: rgba(255,255,255,0.85); font-weight: 700; "
                              "font-size: 10pt; border-bottom: 1px solid rgba(255,255,255,0.05);")
            inner_lay.addWidget(grp)

            if group_id:  # direct nav (Arena)
                item = SidebarItem(group_label, group_id, indent=0)
                item.clicked.connect(self._on_click)
                self._items.append(item)
                inner_lay.addWidget(item)
            else:
                for child_label, child_id in children:
                    item = SidebarItem(child_label, child_id, indent=1)
                    item.clicked.connect(self._on_click)
                    self._items.append(item)
                    inner_lay.addWidget(item)

        inner_lay.addStretch()
        scroll.setWidget(inner); lay.addWidget(scroll)
        self._refresh_active()

    def _on_click(self, page_id: str):
        self._active_id = page_id; self._refresh_active()
        self.page_requested.emit(page_id)

    def _refresh_active(self):
        for item in self._items:
            item.set_active(item.page_id == self._active_id)

    def set_phase(self, phase: float):
        self._phase = phase
        for item in self._items:
            if item._active: item.set_phase(phase)

    def set_active(self, page_id: str):
        self._active_id = page_id; self._refresh_active()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), SURFACE1)
        p.setPen(QPen(QColor(255, 255, 255, 15), 1))
        p.drawLine(self.width() - 1, 0, self.width() - 1, self.height())
        p.end()
```

- [ ] **Step 2: Smoke test**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -c "
import sys; sys.path.insert(0,'.')
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv)
from ui.components.sidebar import Sidebar
s = Sidebar(); s.show()
print('Sidebar OK')
from PySide6.QtCore import QTimer; QTimer.singleShot(500, app.quit); app.exec()
"
```

Expected: `Sidebar OK`

- [ ] **Step 3: Commit**
```bash
cd /Users/jsw/20260406/rl0407
git add ui/components/sidebar.py
git commit -m "feat: sidebar navigation component with glassmorphism"
```

---

## Task 11: GridWorldView

**Files:**
- Create: `ui/visualizations/gridworld_view.py`

- [ ] **Step 1: Write `gridworld_view.py`**

`ui/visualizations/gridworld_view.py`:
```python
from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QRadialGradient

from ui.theme import CYAN, MAGENTA, EMERALD, AMBER, BG, SURFACE1, WHITE_60
from ui.visualizations.painter_utils import (
    draw_glow, draw_arrow, draw_glass_rect, heatmap_color
)


class FloatUp:
    """Animated reward float-up text."""
    def __init__(self, x: float, y: float, text: str, color: QColor):
        self.x = x; self.y = y; self.text = text; self.color = color
        self.life = 1.0  # 1.0 → 0.0

    def tick(self, dt: float = 0.05) -> bool:
        self.y -= 2.0; self.life -= dt
        return self.life > 0


class GridWorldView(QWidget):
    """Renders GridWorld Q-heatmap, policy arrows, agent glow, reward float-ups."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self._n = 4
        self._q_table: np.ndarray | None = None   # (n_states, 4)
        self._agent_state: int = 0
        self._obstacles: set[int] = set()
        self._cliffs: set[int] = set()
        self._goal: int = 0
        self._float_ups: list[FloatUp] = []
        self._phase = 0.0

        self._anim = QTimer(self)
        self._anim.timeout.connect(self._tick_floats)
        self._anim.start(33)

    # ── public API ──────────────────────────────────────────────────────────
    def load_env(self, env) -> None:
        self._n = env.n
        self._obstacles = set(env.obstacles)
        self._cliffs = set(env.cliffs)
        self._goal = env.goal_state
        self._agent_state = env.agent_state
        self.update()

    def update_q(self, q_table: np.ndarray) -> None:
        self._q_table = q_table.copy()
        self.update()

    def update_agent(self, state: int, reward: float) -> None:
        self._agent_state = state
        if abs(reward) > 0.05:
            cx, cy = self._state_center(state)
            color = EMERALD if reward > 0 else MAGENTA
            text = f"+{reward:.2f}" if reward > 0 else f"{reward:.2f}"
            self._float_ups.append(FloatUp(cx, cy, text, color))
        self.update()

    def set_phase(self, phase: float) -> None:
        self._phase = phase; self.update()

    # ── internal ────────────────────────────────────────────────────────────
    def _tick_floats(self):
        self._float_ups = [f for f in self._float_ups if f.tick()]
        if self._float_ups: self.update()

    def _cell_size(self) -> float:
        return min(self.width(), self.height()) / self._n

    def _state_to_row_col(self, s: int) -> tuple[int, int]:
        return s // self._n, s % self._n

    def _state_center(self, s: int) -> tuple[float, float]:
        r, c = self._state_to_row_col(s)
        cs = self._cell_size()
        ox = (self.width() - cs * self._n) / 2
        oy = (self.height() - cs * self._n) / 2
        return ox + c * cs + cs / 2, oy + r * cs + cs / 2

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        cs = self._cell_size()
        ox = (self.width() - cs * self._n) / 2
        oy = (self.height() - cs * self._n) / 2

        for s in range(self._n * self._n):
            r, c = self._state_to_row_col(s)
            x = ox + c * cs; y = oy + r * cs
            rect = QRectF(x + 1, y + 1, cs - 2, cs - 2)

            # Background: heatmap from max Q-value
            if self._q_table is not None and s not in self._obstacles:
                max_q = float(np.max(self._q_table[s]))
                min_q = float(np.min(self._q_table))
                max_all = float(np.max(self._q_table))
                cell_color = heatmap_color(max_q, min_q, max_all)
                cell_color.setAlpha(100)
                p.fillRect(rect, cell_color)

            # Special cells
            if s in self._obstacles:
                p.fillRect(rect, QColor(255, 255, 255, 20))
                p.setPen(QPen(QColor(255, 255, 255, 40), 1))
                p.drawLine(int(x + 2), int(y + 2), int(x + cs - 2), int(y + cs - 2))
                p.drawLine(int(x + cs - 2), int(y + 2), int(x + 2), int(y + cs - 2))
            elif s in self._cliffs:
                p.fillRect(rect, QColor(255, 0, 110, 60))
            elif s == self._goal:
                p.fillRect(rect, QColor(16, 185, 129, 80))
                # Star
                p.setPen(QPen(EMERALD, 1))
                font = QFont(); font.setPointSize(int(cs * 0.3)); p.setFont(font)
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "★")

            # Grid border
            p.setPen(QPen(QColor(255, 255, 255, 15), 1))
            p.drawRect(rect)

            # Policy arrows (best action per state)
            if (self._q_table is not None and s not in self._obstacles
                    and s != self._goal and s not in self._cliffs):
                best_a = int(np.argmax(self._q_table[s]))
                cx = x + cs / 2; cy = y + cs / 2
                arrow_size = cs * 0.28
                draw_arrow(p, cx, cy, best_a, arrow_size, WHITE_60, 0.6)

        # Agent glow
        ax, ay = self._state_center(self._agent_state)
        draw_glow(p, QPointF(ax, ay), cs * 0.35, CYAN)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(CYAN)
        r = cs * 0.18
        p.drawEllipse(QRectF(ax - r, ay - r, 2 * r, 2 * r))

        # Float-up rewards
        font = QFont(); font.setPointSize(9); font.setBold(True); p.setFont(font)
        for fu in self._float_ups:
            alpha = int(fu.life * 220)
            col = QColor(fu.color); col.setAlpha(alpha)
            p.setPen(col)
            p.drawText(QRectF(fu.x - 30, fu.y - 10, 60, 20),
                       Qt.AlignmentFlag.AlignCenter, fu.text)
        p.end()
```

- [ ] **Step 2: Commit**
```bash
cd /Users/jsw/20260406/rl0407
git add ui/visualizations/gridworld_view.py
git commit -m "feat: GridWorldView — Q-heatmap, policy arrows, agent glow, float-up rewards"
```

---

## Task 12: CartPoleView

**Files:**
- Create: `ui/visualizations/cartpole_view.py`

- [ ] **Step 1: Write `cartpole_view.py`**

`ui/visualizations/cartpole_view.py`:
```python
from __future__ import annotations
import math
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QRectF, QPointF, Qt
from PySide6.QtGui import QPainter, QColor, QPen, QLinearGradient, QFont

from ui.theme import CYAN, MAGENTA, EMERALD, AMBER, BG, SURFACE1, WHITE_60
from ui.visualizations.painter_utils import draw_glow


class CartPoleView(QWidget):
    """Renders CartPole physics: cart, pole, track, state bars."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 280)
        self._x = 0.0          # cart position (-2.4 … 2.4)
        self._theta = 0.0      # pole angle (rad)
        self._x_dot = 0.0
        self._theta_dot = 0.0
        self._done = False

    # ── public API ──────────────────────────────────────────────────────────
    def update_state(self, x: float, x_dot: float,
                     theta: float, theta_dot: float, done: bool = False) -> None:
        self._x = x; self._x_dot = x_dot
        self._theta = theta; self._theta_dot = theta_dot
        self._done = done
        self.update()

    # ── paint ───────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        track_y = H * 0.65
        track_x0 = W * 0.1; track_x1 = W * 0.9
        track_w = track_x1 - track_x0

        # Track
        p.setPen(QPen(QColor(255, 255, 255, 40), 2))
        p.drawLine(int(track_x0), int(track_y), int(track_x1), int(track_y))

        # Cart position (normalise -2.4…2.4 → track_x0…track_x1)
        cx = track_x0 + (self._x + 2.4) / 4.8 * track_w
        cart_w = W * 0.12; cart_h = H * 0.08
        cart_rect = QRectF(cx - cart_w / 2, track_y - cart_h, cart_w, cart_h)

        # Cart body
        color = MAGENTA if self._done else CYAN
        p.setPen(QPen(color, 2))
        p.setBrush(QColor(color.red(), color.green(), color.blue(), 40))
        p.drawRoundedRect(cart_rect, 4, 4)

        # Wheels
        wr = cart_h * 0.4
        for wx in [cx - cart_w * 0.3, cx + cart_w * 0.3]:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 60))
            p.drawEllipse(QRectF(wx - wr, track_y - wr, 2 * wr, 2 * wr))

        # Pole
        pole_len = H * 0.35
        pole_px = cx + pole_len * math.sin(self._theta)
        pole_py = (track_y - cart_h) - pole_len * math.cos(self._theta)
        pivot = QPointF(cx, track_y - cart_h)
        tip = QPointF(pole_px, pole_py)

        pole_color = EMERALD if abs(self._theta) < 0.1 else AMBER if abs(self._theta) < 0.15 else MAGENTA
        g = QLinearGradient(pivot, tip)
        g.setColorAt(0, QColor(pole_color.red(), pole_color.green(), pole_color.blue(), 200))
        g.setColorAt(1, QColor(pole_color.red(), pole_color.green(), pole_color.blue(), 80))
        p.setPen(QPen(g, 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(pivot, tip)
        draw_glow(p, tip, 8, pole_color)

        # State bars (bottom strip)
        labels = ["x", "ẋ", "θ", "θ̇"]
        vals = [self._x / 2.4, self._x_dot / 3.0,
                self._theta / 0.2094, self._theta_dot / 3.0]
        bar_h = H * 0.06; bar_y = H * 0.88
        bw = (W - 80) / 4; bpad = 8

        font = QFont(); font.setPointSize(8); p.setFont(font)
        for i, (lbl, v) in enumerate(zip(labels, vals)):
            bx = 40 + i * bw
            # bg
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 15))
            p.drawRoundedRect(QRectF(bx, bar_y, bw - bpad, bar_h), 3, 3)
            # fill
            fill_w = abs(v) * (bw - bpad) * 0.5
            fill_x = bx + (bw - bpad) / 2 if v >= 0 else bx + (bw - bpad) / 2 - fill_w
            bar_color = CYAN if v >= 0 else MAGENTA
            p.setBrush(QColor(bar_color.red(), bar_color.green(), bar_color.blue(), 160))
            p.drawRoundedRect(QRectF(fill_x, bar_y + 1, fill_w, bar_h - 2), 2, 2)
            # label
            p.setPen(WHITE_60)
            p.drawText(QRectF(bx, bar_y - 14, bw - bpad, 14),
                       Qt.AlignmentFlag.AlignCenter, lbl)

        p.end()
```

- [ ] **Step 2: Commit**
```bash
cd /Users/jsw/20260406/rl0407
git add ui/visualizations/cartpole_view.py
git commit -m "feat: CartPoleView — physics renderer with pole glow and state bars"
```

---

## Task 13: MazeView

**Files:**
- Create: `ui/visualizations/maze_view.py`

- [ ] **Step 1: Write `maze_view.py`**

`ui/visualizations/maze_view.py`:
```python
from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QRectF, QPointF, Qt
from PySide6.QtGui import QPainter, QColor, QPen, QFont

from ui.theme import CYAN, MAGENTA, EMERALD, BG, WHITE_60
from ui.visualizations.painter_utils import draw_glow


# direction indices matching Maze.DIRS: 0=N, 1=S, 2=W, 3=E
_DX = [0, 0, -1, 1]
_DY = [-1, 1, 0, 0]


class MazeView(QWidget):
    """Renders Maze walls, BFS solution path trail, agent glow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self._walls: np.ndarray | None = None  # shape (H, W, 4)
        self._rows = 5; self._cols = 5
        self._agent_pos: tuple[int, int] = (0, 0)
        self._solution: list[tuple[int, int]] = []
        self._visited: list[tuple[int, int]] = []

    # ── public API ──────────────────────────────────────────────────────────
    def load_maze(self, maze) -> None:
        self._walls = maze.walls.copy()
        self._rows = maze.rows; self._cols = maze.cols
        self._solution = maze.bfs_path() if hasattr(maze, 'bfs_path') else []
        self._agent_pos = (0, 0)
        self._visited = [(0, 0)]
        self.update()

    def update_agent(self, pos: tuple[int, int]) -> None:
        self._agent_pos = pos
        if pos not in self._visited:
            self._visited.append(pos)
        self.update()

    def show_solution(self, show: bool = True) -> None:
        self._show_solution = show; self.update()

    # ── paint ───────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        if self._walls is None:
            p.end(); return

        W, H = self.width(), self.height()
        cs = min(W / self._cols, H / self._rows) * 0.9
        ox = (W - cs * self._cols) / 2
        oy = (H - cs * self._rows) / 2
        wall_w = max(2, cs * 0.08)

        def cell_rect(r, c):
            return QRectF(ox + c * cs, oy + r * cs, cs, cs)

        # Visited path
        for r, c in self._visited:
            rect = cell_rect(r, c)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 212, 255, 18))
            p.drawRect(rect)

        # Solution path
        if hasattr(self, '_show_solution') and self._show_solution and self._solution:
            for r, c in self._solution:
                rect = cell_rect(r, c)
                p.setBrush(QColor(16, 185, 129, 35))
                p.drawRect(rect)

        # Goal
        gr, gc = self._rows - 1, self._cols - 1
        goal_rect = cell_rect(gr, gc)
        p.setBrush(QColor(16, 185, 129, 60))
        p.drawRect(goal_rect)
        p.setPen(EMERALD)
        font = QFont(); font.setPointSize(int(cs * 0.3)); p.setFont(font)
        p.drawText(goal_rect, Qt.AlignmentFlag.AlignCenter, "★")

        # Walls
        p.setPen(QPen(QColor(255, 255, 255, 80), wall_w,
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
        for r in range(self._rows):
            for c in range(self._cols):
                x0 = ox + c * cs; y0 = oy + r * cs
                x1 = x0 + cs;     y1 = y0 + cs
                walls = self._walls[r, c]  # [N, S, W, E]
                if walls[0]: p.drawLine(int(x0), int(y0), int(x1), int(y0))  # N
                if walls[1]: p.drawLine(int(x0), int(y1), int(x1), int(y1))  # S
                if walls[2]: p.drawLine(int(x0), int(y0), int(x0), int(y1))  # W
                if walls[3]: p.drawLine(int(x1), int(y0), int(x1), int(y1))  # E

        # Outer border
        p.setPen(QPen(QColor(255, 255, 255, 120), wall_w))
        p.drawRect(QRectF(ox, oy, cs * self._cols, cs * self._rows))

        # Agent
        ar, ac = self._agent_pos
        ax = ox + ac * cs + cs / 2; ay = oy + ar * cs + cs / 2
        draw_glow(p, QPointF(ax, ay), cs * 0.3, CYAN)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(CYAN)
        r = cs * 0.18
        p.drawEllipse(QRectF(ax - r, ay - r, 2 * r, 2 * r))

        p.end()
```

- [ ] **Step 2: Commit**
```bash
cd /Users/jsw/20260406/rl0407
git add ui/visualizations/maze_view.py
git commit -m "feat: MazeView — wall rendering, visited path, solution trail, agent glow"
```

---

## Task 14: Controllers

**Files:**
- Create: `controllers/training_controller.py`
- Create: `controllers/concept_controller.py`
- Create: `controllers/arena_controller.py`

- [ ] **Step 1: Write `training_controller.py`**

`controllers/training_controller.py`:
```python
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

    episode_done = Signal(int, float)         # (episode, total_reward)
    step_done = Signal(int, float, object)    # (state, reward, env)
    agent_moved = Signal(int, float)          # (state, reward) — GridWorld/CartPole
    maze_moved = Signal(tuple, float)         # ((r,c), reward) — Maze
    cartpole_state = Signal(float, float, float, float, bool)  # x, xdot, th, thdot, done
    training_speed = Signal(float)            # current steps/sec

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

    # ── public ───────────────────────────────────────────────────────────────
    def setup(self, env_type: str, env_cfg: dict, algo_type: str, algo_cfg: dict):
        self._env_type = env_type
        if env_type == "gridworld":
            self._env = GridWorld(**env_cfg)
            n_states = self._env.n * self._env.n
            if algo_type == "qlearning":
                self._agent = QLearning(n_states=n_states, n_actions=4, **algo_cfg)
            else:
                self._agent = REINFORCE(
                    policy=LinearPolicy(n_states, 4), **algo_cfg)
        elif env_type == "cartpole":
            self._env = CartPole(**env_cfg)
            n_states = 10 ** 4
            if algo_type == "qlearning":
                self._agent = QLearning(n_states=n_states, n_actions=2, **algo_cfg)
            else:
                self._agent = REINFORCE(
                    policy=LinearPolicy(n_states, 2), **algo_cfg)
        elif env_type == "maze":
            self._env = Maze(**env_cfg)
            n_states = self._env.rows * self._env.cols
            if algo_type == "qlearning":
                self._agent = QLearning(n_states=n_states, n_actions=4, **algo_cfg)
            else:
                self._agent = REINFORCE(
                    policy=LinearPolicy(n_states, 4), **algo_cfg)
        self._reset_episode()

    def set_speed(self, steps_per_sec: int):
        self._speed_ms = max(10, 1000 // steps_per_sec)
        if self._timer.isActive():
            self._timer.setInterval(self._speed_ms)

    def start(self):
        if self._env and self._agent:
            self._timer.start(self._speed_ms)

    def stop(self):
        self._timer.stop()

    def reset(self):
        self.stop()
        if self._env: self._env.reset()
        self._episode = 0
        self._reset_episode()

    def get_q_table(self) -> np.ndarray | None:
        if hasattr(self._agent, 'q_table'):
            return self._agent.q_table
        return None

    # ── internal ─────────────────────────────────────────────────────────────
    def _reset_episode(self):
        if self._env is None: return
        state = self._env.reset()
        self._current_state = state
        self._states = []; self._actions = []; self._rewards = []
        self._episode_reward = 0.0

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
            self.cartpole_state.emit(*obs, done)
        else:
            ns, reward, done = self._env.step(a)
            self.agent_moved.emit(ns, reward)

        if hasattr(self._agent, 'update'):
            self._agent.update(s, a, reward, ns, done)

        self._states.append(s); self._actions.append(a); self._rewards.append(reward)
        self._episode_reward += reward
        self._current_state = ns

        if done or self._steps > 500:
            if hasattr(self._agent, 'finish_episode'):
                self._agent.finish_episode(self._states, self._actions, self._rewards)
            self._episode += 1
            self.episode_done.emit(self._episode, self._episode_reward)
            self._reset_episode()
        self._steps += 1

    @property
    def env(self):
        return self._env

    @property
    def agent(self):
        return self._agent
```

- [ ] **Step 2: Write `concept_controller.py`**

`controllers/concept_controller.py`:
```python
from __future__ import annotations
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer


class ConceptController(QObject):
    """Drives animated concept demos (MDP transitions, Bellman backup, etc.)."""

    phase_updated = Signal(float)   # 0.0 … 1.0 animation phase
    demo_state = Signal(dict)       # arbitrary payload for concept widgets

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._phase = 0.0
        self._speed = 0.01

    def start(self, speed: float = 0.01):
        self._speed = speed
        self._timer.start(33)

    def stop(self):
        self._timer.stop()

    def _tick(self):
        self._phase = (self._phase + self._speed) % 1.0
        self.phase_updated.emit(self._phase)
```

- [ ] **Step 3: Write `arena_controller.py`**

`controllers/arena_controller.py`:
```python
from __future__ import annotations
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer

from core.environments.gridworld import GridWorld
from core.algorithms.qlearning import QLearning
from core.algorithms.reinforce import REINFORCE, LinearPolicy


class ArenaController(QObject):
    """Runs Q-Learning vs REINFORCE head-to-head on GridWorld."""

    step_done = Signal(int, float, float)   # (episode, ql_reward, rf_reward)
    q_table_updated = Signal(np.ndarray)    # QLearning Q-table
    rf_policy_updated = Signal(np.ndarray) # REINFORCE policy weights

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
        self._ql_ep_states: list = []; self._ql_ep_actions: list = []
        self._ql_ep_rewards: list = []; self._ql_ep_reward = 0.0
        self._rf_ep_states: list = []; self._rf_ep_actions: list = []
        self._rf_ep_rewards: list = []; self._rf_ep_reward = 0.0
        self._ql_state = self._env_ql.reset()
        self._rf_state = self._env_rf.reset()
        self._ql_steps = 0; self._rf_steps = 0

    def start(self):
        self._timer.start(50)

    def stop(self):
        self._timer.stop()

    def reset(self):
        self.stop()
        self._episode = 0
        self._ql_state = self._env_ql.reset()
        self._rf_state = self._env_rf.reset()
        self._ql_ep_reward = 0.0; self._rf_ep_reward = 0.0
        self._ql_steps = 0; self._rf_steps = 0
        self._ql_ep_states = []; self._ql_ep_actions = []; self._ql_ep_rewards = []
        self._rf_ep_states = []; self._rf_ep_actions = []; self._rf_ep_rewards = []

    def _step(self):
        # QLearning step
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
            self._ql_state = self._env_ql.reset(); self._rf_state = self._env_rf.reset()
            self._ql_ep_reward = 0.0; self._rf_ep_reward = 0.0
            self._ql_steps = 0; self._rf_steps = 0
            self._rf_ep_states = []; self._rf_ep_actions = []; self._rf_ep_rewards = []
```

- [ ] **Step 4: Commit**
```bash
cd /Users/jsw/20260406/rl0407
git add controllers/training_controller.py controllers/concept_controller.py controllers/arena_controller.py
git commit -m "feat: training, concept, and arena controllers"
```

---

## Task 15: Concept Views — RL Basics & Value-Based

**Files:**
- Create: `ui/views/concepts/rl_basics_view.py`
- Create: `ui/views/concepts/value_based_view.py`

- [ ] **Step 1: Write `rl_basics_view.py`**

`ui/views/concepts/rl_basics_view.py`:
```python
from __future__ import annotations
import math
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import QRectF, QPointF, Qt, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient, QRadialGradient

from ui.theme import (CYAN, MAGENTA, VIOLET, EMERALD, AMBER,
                      BG, SURFACE1, WHITE_60, WHITE_40, pulse, lerp_color)
from ui.visualizations.painter_utils import draw_glow, draw_glass_rect


class AgentEnvLoopWidget(QWidget):
    """Animated agent–environment loop diagram."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self._phase = 0.0

    def set_phase(self, phase: float):
        self._phase = phase; self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        # Two boxes: Agent (left) and Environment (right)
        bw = W * 0.28; bh = H * 0.35
        ag_x = W * 0.12; ag_y = (H - bh) / 2
        env_x = W * 0.60; env_y = (H - bh) / 2

        # Box fill
        draw_glass_rect(p, QRectF(ag_x, ag_y, bw, bh))
        draw_glass_rect(p, QRectF(env_x, env_y, bw, bh))

        # Labels
        font = QFont(); font.setPointSize(11); font.setBold(True); p.setFont(font)
        p.setPen(CYAN)
        p.drawText(QRectF(ag_x, ag_y, bw, bh), Qt.AlignmentFlag.AlignCenter, "Agent")
        p.setPen(EMERALD)
        p.drawText(QRectF(env_x, env_y, bw, bh), Qt.AlignmentFlag.AlignCenter, "Environ\nment")

        # Arrows with animated dot
        arrow_y_top = ag_y - H * 0.12   # action arrow above
        arrow_y_bot = ag_y + bh + H * 0.05  # reward+state arrow below

        cx_ag_right = ag_x + bw; cx_env_left = env_x
        mid_x = (cx_ag_right + cx_env_left) / 2

        # Action arrow: Agent → Env (top, cyan)
        p.setPen(QPen(CYAN, 2))
        p.drawLine(int(cx_ag_right), int(ag_y + bh * 0.35),
                   int(cx_env_left), int(env_y + bh * 0.35))
        # arrowhead
        p.setBrush(CYAN); p.setPen(Qt.PenStyle.NoPen)
        self._draw_arrowhead(p, cx_env_left, env_y + bh * 0.35, right=True)

        label_font = QFont(); label_font.setPointSize(8); p.setFont(label_font)
        p.setPen(CYAN)
        p.drawText(QRectF(cx_ag_right, ag_y + bh * 0.35 - 18, mid_x - cx_ag_right, 16),
                   Qt.AlignmentFlag.AlignCenter, "Action aₜ")

        # Animated dot on action arrow
        t_action = (self._phase * 1.5) % 1.0
        dot_x = cx_ag_right + t_action * (cx_env_left - cx_ag_right)
        draw_glow(p, QPointF(dot_x, ag_y + bh * 0.35), 5, CYAN)

        # State + Reward arrow: Env → Agent (bottom, magenta+amber)
        p.setPen(QPen(MAGENTA, 2))
        p.drawLine(int(cx_env_left), int(env_y + bh * 0.65),
                   int(cx_ag_right), int(ag_y + bh * 0.65))
        p.setBrush(MAGENTA); p.setPen(Qt.PenStyle.NoPen)
        self._draw_arrowhead(p, cx_ag_right, ag_y + bh * 0.65, right=False)

        p.setPen(MAGENTA)
        p.setFont(label_font)
        p.drawText(QRectF(cx_ag_right, ag_y + bh * 0.65 + 2, mid_x - cx_ag_right, 16),
                   Qt.AlignmentFlag.AlignCenter, "State sₜ₊₁, Reward rₜ")

        t_state = ((self._phase * 1.5) + 0.5) % 1.0
        dot_x2 = cx_env_left + t_state * (cx_ag_right - cx_env_left)
        draw_glow(p, QPointF(dot_x2, ag_y + bh * 0.65), 5, MAGENTA)

        p.end()

    def _draw_arrowhead(self, p: QPainter, tip_x: float, tip_y: float, right: bool):
        d = 8
        if right:
            pts = [QPointF(tip_x, tip_y),
                   QPointF(tip_x - d, tip_y - d / 2),
                   QPointF(tip_x - d, tip_y + d / 2)]
        else:
            pts = [QPointF(tip_x, tip_y),
                   QPointF(tip_x + d, tip_y - d / 2),
                   QPointF(tip_x + d, tip_y + d / 2)]
        from PySide6.QtGui import QPolygonF
        p.drawPolygon(QPolygonF(pts))


class MDPWidget(QWidget):
    """MDP states with transition arrows and reward labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self._phase = 0.0

    def set_phase(self, phase: float):
        self._phase = phase; self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        centers = [
            QPointF(W * 0.15, H * 0.5),
            QPointF(W * 0.4, H * 0.3),
            QPointF(W * 0.65, H * 0.5),
            QPointF(W * 0.88, H * 0.5),
        ]
        labels = ["S₀", "S₁", "S₂", "S₃(G)"]
        colors = [CYAN, VIOLET, AMBER, EMERALD]
        r_state = 22

        # Draw transitions
        transitions = [(0, 1, "a=0\nr=−0.01"), (0, 2, "a=1\nr=−0.01"),
                       (1, 2, "a=1\nr=−0.01"), (2, 3, "a=1\nr=+1.0")]
        p.setPen(QPen(WHITE_40, 1.5))
        font = QFont(); font.setPointSize(7); p.setFont(font)
        for s_from, s_to, lbl in transitions:
            c1 = centers[s_from]; c2 = centers[s_to]
            p.drawLine(c1, c2)
            mid = QPointF((c1.x() + c2.x()) / 2, (c1.y() + c2.y()) / 2 - 12)
            p.setPen(WHITE_40)
            p.drawText(QRectF(mid.x() - 25, mid.y() - 10, 50, 20),
                       Qt.AlignmentFlag.AlignCenter, lbl)

        # Animated traversal dot
        t = self._phase % 1.0
        n = len(transitions)
        seg = int(t * n); frac = (t * n) % 1.0
        s_from, s_to, _ = transitions[min(seg, n - 1)]
        c1 = centers[s_from]; c2 = centers[s_to]
        dot = QPointF(c1.x() + frac * (c2.x() - c1.x()),
                      c1.y() + frac * (c2.y() - c1.y()))
        draw_glow(p, dot, 6, CYAN)

        # Draw state circles
        font2 = QFont(); font2.setPointSize(9); font2.setBold(True)
        for i, (c, lbl, col) in enumerate(zip(centers, labels, colors)):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(col.red(), col.green(), col.blue(), 40))
            p.drawEllipse(QRectF(c.x() - r_state, c.y() - r_state,
                                 2 * r_state, 2 * r_state))
            p.setPen(QPen(col, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(c.x() - r_state, c.y() - r_state,
                                 2 * r_state, 2 * r_state))
            p.setPen(col); p.setFont(font2)
            p.drawText(QRectF(c.x() - r_state, c.y() - r_state,
                               2 * r_state, 2 * r_state),
                       Qt.AlignmentFlag.AlignCenter, lbl)

        p.end()


class DiscountWidget(QWidget):
    """Shows geometric discount: Gₜ = r + γr + γ²r + ..."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self._gamma = 0.9

    def set_gamma(self, gamma: float):
        self._gamma = gamma; self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        n = 8
        max_val = 1.0
        bar_w = (W - 80) / n; bar_max_h = H * 0.6

        font = QFont(); font.setPointSize(8); p.setFont(font)
        for i in range(n):
            val = self._gamma ** i
            bh = val / max_val * bar_max_h
            bx = 40 + i * bar_w
            by = H * 0.75 - bh

            color = lerp_color(CYAN, MAGENTA, i / (n - 1))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(color.red(), color.green(), color.blue(), 160))
            p.drawRoundedRect(QRectF(bx + 2, by, bar_w - 4, bh), 3, 3)

            p.setPen(WHITE_60)
            p.drawText(QRectF(bx, H * 0.78, bar_w, 20),
                       Qt.AlignmentFlag.AlignCenter, f"γ{i}")
            p.drawText(QRectF(bx, by - 18, bar_w, 16),
                       Qt.AlignmentFlag.AlignCenter, f"{val:.2f}")

        # Title
        font2 = QFont(); font2.setPointSize(9); font2.setBold(True); p.setFont(font2)
        p.setPen(WHITE_60)
        p.drawText(QRectF(0, 4, W, 20), Qt.AlignmentFlag.AlignCenter,
                   f"Discount Factor γ = {self._gamma:.2f}")
        p.end()


class RLBasicsView(QWidget):
    phase_updated = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20); lay.setSpacing(16)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none; background:transparent;}")
        inner = QWidget(); inner_lay = QVBoxLayout(inner)
        inner_lay.setSpacing(20); inner_lay.setContentsMargins(0, 0, 0, 0)

        def section(title):
            lbl = QLabel(title)
            lbl.setStyleSheet("color:#00D4FF; font-size:12pt; font-weight:700; "
                              "border-bottom:1px solid rgba(0,212,255,0.25); padding-bottom:4px;")
            return lbl

        inner_lay.addWidget(section("Agent–Environment Loop"))
        self._loop = AgentEnvLoopWidget()
        inner_lay.addWidget(self._loop)

        inner_lay.addWidget(section("Markov Decision Process (MDP)"))
        self._mdp = MDPWidget()
        inner_lay.addWidget(self._mdp)

        inner_lay.addWidget(section("Discount Factor"))
        self._discount = DiscountWidget()
        inner_lay.addWidget(self._discount)

        inner_lay.addStretch()
        scroll.setWidget(inner); lay.addWidget(scroll)

    def set_phase(self, phase: float):
        self._loop.set_phase(phase)
        self._mdp.set_phase(phase)
```

- [ ] **Step 2: Write `value_based_view.py`**

`ui/views/concepts/value_based_view.py`:
```python
from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainter, QColor, QPen, QFont

from ui.theme import (CYAN, MAGENTA, VIOLET, EMERALD, AMBER,
                      BG, SURFACE1, WHITE_60, WHITE_40, lerp_color)
from ui.visualizations.painter_utils import draw_glow, draw_glass_rect, heatmap_color


class QTableWidget(QWidget):
    """Live mini Q-table grid (state rows × action cols)."""

    def __init__(self, n_states: int = 4, n_actions: int = 4, parent=None):
        super().__init__(parent)
        self._n_states = min(n_states, 16)
        self._n_actions = n_actions
        self._q = np.zeros((self._n_states, self._n_actions))
        self.setMinimumHeight(180)

    def update_q(self, q: np.ndarray):
        self._q = q[:self._n_states].copy(); self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        n_rows = self._n_states; n_cols = self._n_actions
        col_w = (W - 60) / n_cols; row_h = (H - 30) / n_rows
        ox = 60; oy = 30

        # Column headers
        headers = ["↑Up", "↓Dn", "←Lt", "→Rt"][:n_cols]
        font = QFont(); font.setPointSize(8); p.setFont(font)
        p.setPen(CYAN)
        for j, h in enumerate(headers):
            p.drawText(QRectF(ox + j * col_w, 4, col_w, 22),
                       Qt.AlignmentFlag.AlignCenter, h)

        min_q = float(np.min(self._q)); max_q = float(np.max(self._q))
        if max_q == min_q: max_q = min_q + 1

        for i in range(n_rows):
            # Row label
            p.setPen(WHITE_40)
            p.drawText(QRectF(0, oy + i * row_h, 56, row_h),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                       f"S{i}")
            for j in range(n_cols):
                val = self._q[i, j]
                cell = QRectF(ox + j * col_w + 1, oy + i * row_h + 1,
                               col_w - 2, row_h - 2)
                c = heatmap_color(val, min_q, max_q); c.setAlpha(150)
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(c)
                p.drawRect(cell)
                p.setPen(WHITE_60)
                p.drawText(cell, Qt.AlignmentFlag.AlignCenter, f"{val:.2f}")
        p.end()


class BellmanWidget(QWidget):
    """Animated Bellman backup diagram."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self._phase = 0.0

    def set_phase(self, phase: float):
        self._phase = phase; self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        # Root state
        root = (W / 2, H * 0.2)
        children = [(W * 0.25, H * 0.7), (W * 0.5, H * 0.7), (W * 0.75, H * 0.7)]
        rewards = ["-0.01", "+1.0", "-10.0"]
        colors = [CYAN, EMERALD, MAGENTA]

        # Animated highlight which action is "selected"
        sel = int(self._phase * 3) % 3
        bright = (self._phase * 3) % 1.0

        for i, (cx, cy) in enumerate(children):
            c = colors[i] if i == sel else WHITE_40
            p.setPen(QPen(c, 2 if i == sel else 1))
            p.drawLine(int(root[0]), int(root[1] + 18), int(cx), int(cy - 18))
            # Reward label
            p.setPen(c)
            font = QFont(); font.setPointSize(8); p.setFont(font)
            p.drawText(QRectF(cx - 20, (root[1] + cy) / 2 - 8, 40, 16),
                       Qt.AlignmentFlag.AlignCenter, f"r={rewards[i]}")
            # Child node
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 60 if i == sel else 20))
            p.setPen(QPen(c, 2))
            p.drawEllipse(QRectF(cx - 18, cy - 18, 36, 36))
            p.setPen(c)
            font2 = QFont(); font2.setPointSize(8); p.setFont(font2)
            p.drawText(QRectF(cx - 18, cy - 18, 36, 36),
                       Qt.AlignmentFlag.AlignCenter, f"V(s')")

        # Root
        br = pulse(self._phase, 0.7, 1.0)
        rc = QColor(int(CYAN.red() * br), int(CYAN.green() * br), int(CYAN.blue() * br))
        p.setBrush(QColor(rc.red(), rc.green(), rc.blue(), 80))
        p.setPen(QPen(rc, 2))
        p.drawEllipse(QRectF(root[0] - 22, root[1] - 18, 44, 36))
        p.setPen(rc)
        font3 = QFont(); font3.setPointSize(9); font3.setBold(True); p.setFont(font3)
        p.drawText(QRectF(root[0] - 22, root[1] - 18, 44, 36),
                   Qt.AlignmentFlag.AlignCenter, "Q(s,a)")

        # Max arrow annotation
        p.setPen(AMBER)
        font4 = QFont(); font4.setPointSize(8); p.setFont(font4)
        p.drawText(QRectF(W * 0.05, H * 0.85, W * 0.9, 20),
                   Qt.AlignmentFlag.AlignCenter,
                   "Q(s,a) = r + γ·max Q(s',a')  (Bellman equation)")

        p.end()


class EpsilonGreedyWidget(QWidget):
    """Visualises ε-greedy action selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self._epsilon = 0.2

    def set_epsilon(self, eps: float):
        self._epsilon = eps; self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        # Pie: exploit (1-ε) CYAN vs explore ε MAGENTA
        import math
        cx = W / 2; cy = H * 0.5; r = min(W, H) * 0.3
        eps = self._epsilon

        # Explore slice
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(MAGENTA.red(), MAGENTA.green(), MAGENTA.blue(), 180))
        p.drawPie(QRectF(cx - r, cy - r, 2 * r, 2 * r),
                  90 * 16, int(eps * 360 * 16))

        # Exploit slice
        p.setBrush(QColor(CYAN.red(), CYAN.green(), CYAN.blue(), 180))
        p.drawPie(QRectF(cx - r, cy - r, 2 * r, 2 * r),
                  int((90 + eps * 360) * 16), int((1 - eps) * 360 * 16))

        font = QFont(); font.setPointSize(9); p.setFont(font)
        p.setPen(MAGENTA)
        p.drawText(QRectF(cx - r - 80, cy - 10, 70, 20),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"Explore ε={eps:.2f}")
        p.setPen(CYAN)
        p.drawText(QRectF(cx + r + 10, cy - 10, 100, 20),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"Exploit {1-eps:.2f}")
        p.end()


class ValueBasedView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20); lay.setSpacing(16)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none; background:transparent;}")
        inner = QWidget(); inner_lay = QVBoxLayout(inner)
        inner_lay.setSpacing(20); inner_lay.setContentsMargins(0, 0, 0, 0)

        def section(title):
            lbl = QLabel(title)
            lbl.setStyleSheet("color:#00D4FF; font-size:12pt; font-weight:700; "
                              "border-bottom:1px solid rgba(0,212,255,0.25); padding-bottom:4px;")
            return lbl

        inner_lay.addWidget(section("Q-Table (Live)"))
        self._qt = QTableWidget(n_states=16, n_actions=4)
        inner_lay.addWidget(self._qt)

        inner_lay.addWidget(section("Bellman Backup"))
        self._bellman = BellmanWidget()
        inner_lay.addWidget(self._bellman)

        inner_lay.addWidget(section("ε-Greedy Exploration"))
        self._eps_widget = EpsilonGreedyWidget()
        inner_lay.addWidget(self._eps_widget)

        inner_lay.addStretch()
        scroll.setWidget(inner); lay.addWidget(scroll)

    def set_phase(self, phase: float):
        self._bellman.set_phase(phase)

    def update_q(self, q: np.ndarray):
        self._qt.update_q(q)
```

- [ ] **Step 3: Commit**
```bash
cd /Users/jsw/20260406/rl0407
git add ui/views/concepts/rl_basics_view.py ui/views/concepts/value_based_view.py
git commit -m "feat: concept views — RL basics and value-based widgets"
```

---

## Task 16: Concept Views — Policy-Based & Applications

**Files:**
- Create: `ui/views/concepts/policy_based_view.py`
- Create: `ui/views/concepts/applications_view.py`

- [ ] **Step 1: Write `policy_based_view.py`**

`ui/views/concepts/policy_based_view.py`:
```python
from __future__ import annotations
import math
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import QRectF, QPointF, Qt
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient

from ui.theme import (CYAN, MAGENTA, VIOLET, EMERALD, AMBER,
                      BG, WHITE_60, WHITE_40, pulse)
from ui.visualizations.painter_utils import draw_glow, draw_glass_rect


class PolicyWidget(QWidget):
    """Stochastic policy probabilities as bar chart."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self._probs = np.array([0.1, 0.5, 0.3, 0.1])
        self._labels = ["↑", "↓", "←", "→"]

    def set_probs(self, probs: np.ndarray):
        self._probs = np.clip(probs, 0, 1); self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()
        n = len(self._probs)
        bw = (W - 60) / n; bpad = 8
        max_h = H * 0.6; oy = H * 0.15

        font = QFont(); font.setPointSize(9); p.setFont(font)
        for i, (prob, lbl) in enumerate(zip(self._probs, self._labels)):
            bx = 30 + i * bw
            bh = prob * max_h
            by = oy + max_h - bh

            color = VIOLET if i == int(np.argmax(self._probs)) else CYAN
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(color.red(), color.green(), color.blue(), 160))
            p.drawRoundedRect(QRectF(bx + bpad / 2, by, bw - bpad, bh), 4, 4)

            p.setPen(WHITE_60)
            p.drawText(QRectF(bx, oy + max_h + 4, bw, 18),
                       Qt.AlignmentFlag.AlignCenter, lbl)
            p.drawText(QRectF(bx, by - 16, bw, 14),
                       Qt.AlignmentFlag.AlignCenter, f"{prob:.2f}")

        p.setPen(WHITE_60)
        font2 = QFont(); font2.setPointSize(8); p.setFont(font2)
        p.drawText(QRectF(0, H - 18, W, 16), Qt.AlignmentFlag.AlignCenter,
                   "π(a|s) — softmax policy probabilities")
        p.end()


class PolicyGradientWidget(QWidget):
    """Animated gradient ascent on policy space."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self._phase = 0.0
        self._reward_history: list[float] = []

    def set_phase(self, phase: float):
        self._phase = phase; self.update()

    def add_reward(self, r: float):
        self._reward_history.append(r)
        if len(self._reward_history) > 60:
            self._reward_history.pop(0)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), BG)

        W, H = self.width(), self.height()

        if len(self._reward_history) < 2:
            font = QFont(); font.setPointSize(9); p.setFont(font)
            p.setPen(WHITE_40)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Run Arena → REINFORCE to see gradient ascent")
            p.end(); return

        # Simple line chart of returns
        n = len(self._reward_history)
        chart_x0 = 40; chart_x1 = W - 20
        chart_y0 = H * 0.1; chart_y1 = H * 0.85
        cw = chart_x1 - chart_x0; ch = chart_y1 - chart_y0

        min_r = min(self._reward_history); max_r = max(self._reward_history)
        span = max(max_r - min_r, 1.0)

        def to_screen(i, v):
            sx = chart_x0 + i / max(n - 1, 1) * cw
            sy = chart_y1 - (v - min_r) / span * ch
            return sx, sy

        # Gradient fill
        g = QLinearGradient(0, chart_y0, 0, chart_y1)
        g.setColorAt(0, QColor(124, 58, 237, 60))
        g.setColorAt(1, QColor(124, 58, 237, 0))

        from PySide6.QtGui import QPolygonF
        poly_pts = [QPointF(chart_x0, chart_y1)]
        for i, v in enumerate(self._reward_history):
            sx, sy = to_screen(i, v); poly_pts.append(QPointF(sx, sy))
        poly_pts.append(QPointF(chart_x1, chart_y1))
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(g)
        p.drawPolygon(QPolygonF(poly_pts))

        # Line
        p.setPen(QPen(VIOLET, 2))
        for i in range(1, n):
            x1, y1 = to_screen(i - 1, self._reward_history[i - 1])
            x2, y2 = to_screen(i, self._reward_history[i])
            p.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Current dot
        lx, ly = to_screen(n - 1, self._reward_history[-1])
        draw_glow(p, QPointF(lx, ly), 6, VIOLET)

        font = QFont(); font.setPointSize(8); p.setFont(font)
        p.setPen(WHITE_60)
        p.drawText(QRectF(0, H - 18, W, 16), Qt.AlignmentFlag.AlignCenter,
                   "REINFORCE — Episode Returns (Policy Gradient)")
        p.end()


class PolicyBasedView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20); lay.setSpacing(16)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none; background:transparent;}")
        inner = QWidget(); inner_lay = QVBoxLayout(inner)
        inner_lay.setSpacing(20); inner_lay.setContentsMargins(0, 0, 0, 0)

        def section(title):
            lbl = QLabel(title)
            lbl.setStyleSheet("color:#7C3AED; font-size:12pt; font-weight:700; "
                              "border-bottom:1px solid rgba(124,58,237,0.35); padding-bottom:4px;")
            return lbl

        inner_lay.addWidget(section("Stochastic Policy π(a|s)"))
        self._policy = PolicyWidget()
        inner_lay.addWidget(self._policy)

        inner_lay.addWidget(section("Policy Gradient — Episode Returns"))
        self._pg = PolicyGradientWidget()
        inner_lay.addWidget(self._pg)

        inner_lay.addWidget(section("REINFORCE Algorithm"))
        txt = QLabel(
            "1. Run episode under current policy π_θ\n"
            "2. Compute discounted returns Gₜ = Σ γᵏ rₜ₊ₖ\n"
            "3. Normalize returns: Ĝₜ = (Gₜ − μ) / (σ + ε)\n"
            "4. Update θ ← θ + α Σₜ Ĝₜ ∇θ log π_θ(aₜ|sₜ)\n"
            "5. Repeat until convergence"
        )
        txt.setStyleSheet("color: rgba(255,255,255,0.7); font-family: monospace; "
                          "font-size: 10pt; padding: 12px; "
                          "background: rgba(255,255,255,0.05); border-radius: 8px;")
        inner_lay.addWidget(txt)

        inner_lay.addStretch()
        scroll.setWidget(inner); lay.addWidget(scroll)

    def set_phase(self, phase: float):
        self._pg.set_phase(phase)

    def update_probs(self, probs: np.ndarray):
        self._policy.set_probs(probs)

    def add_reward(self, r: float):
        self._pg.add_reward(r)
```

- [ ] **Step 2: Write `applications_view.py`**

`ui/views/concepts/applications_view.py`:
```python
from __future__ import annotations
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient

from ui.theme import CYAN, MAGENTA, VIOLET, EMERALD, AMBER, BG, SURFACE1, WHITE_60
from ui.visualizations.painter_utils import draw_glass_rect


class ApplicationCard(QWidget):
    """Single RL application card with title, description, key result."""

    def __init__(self, title: str, year: str, desc: str,
                 result: str, color: QColor, parent=None):
        super().__init__(parent)
        self._title = title; self._year = year
        self._desc = desc; self._result = result; self._color = color
        self.setMinimumHeight(120)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        draw_glass_rect(p, QRectF(4, 4, self.width() - 8, self.height() - 8))

        W, H = self.width(), self.height()
        # Left accent bar
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self._color)
        p.drawRoundedRect(QRectF(4, 4, 4, H - 8), 2, 2)

        # Title
        font = QFont(); font.setPointSize(11); font.setBold(True); p.setFont(font)
        p.setPen(self._color)
        p.drawText(QRectF(20, 10, W - 100, 28), Qt.AlignmentFlag.AlignVCenter, self._title)

        # Year badge
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(self._color.red(), self._color.green(), self._color.blue(), 40))
        p.drawRoundedRect(QRectF(W - 70, 10, 60, 22), 4, 4)
        font2 = QFont(); font2.setPointSize(8); p.setFont(font2)
        p.setPen(self._color)
        p.drawText(QRectF(W - 70, 10, 60, 22), Qt.AlignmentFlag.AlignCenter, self._year)

        # Description
        font3 = QFont(); font3.setPointSize(9); p.setFont(font3)
        p.setPen(WHITE_60)
        p.drawText(QRectF(20, 44, W - 30, H - 80), Qt.AlignmentFlag.AlignTop, self._desc)

        # Result
        font4 = QFont(); font4.setPointSize(8); font4.setBold(True); p.setFont(font4)
        p.setPen(self._color)
        p.drawText(QRectF(20, H - 28, W - 30, 22), Qt.AlignmentFlag.AlignVCenter,
                   f"★  {self._result}")
        p.end()


APPLICATIONS = [
    ("DQN — Atari", "2013", "DeepMind trains CNN+Q-Learning on raw Atari pixels.\n"
     "49 games, superhuman on 29.",
     "Human-level control from pixels", CYAN),
    ("AlphaGo / AlphaZero", "2016–17",
     "Combines MCTS with policy+value networks. Self-play from scratch.",
     "Defeated world champion (Lee Sedol 4–1)", VIOLET),
    ("OpenAI Five — Dota 2", "2019",
     "5-agent team, multi-agent RL, 180 years of self-play/day.",
     "World championship defeat — OG team", EMERALD),
    ("AlphaStar — StarCraft II", "2019",
     "Multiagent population-based RL, 200 years training, 500 APM.",
     "Grandmaster level in all three races", AMBER),
    ("Robotics — Dexterous Manipulation", "2019",
     "OpenAI Dactyl: domain randomization + PPO on Shadow Hand.",
     "Solved Rubik's cube one-handed", MAGENTA),
    ("ChatGPT — RLHF", "2022",
     "Reinforcement Learning from Human Feedback fine-tunes LLMs.\n"
     "Reward model from human preference labels.",
     "State-of-the-art conversational AI", QColor(255, 200, 50)),
]


class ApplicationsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20); lay.setSpacing(12)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none; background:transparent;}")
        inner = QWidget(); inner_lay = QVBoxLayout(inner)
        inner_lay.setSpacing(12); inner_lay.setContentsMargins(0, 0, 0, 0)

        hdr = QLabel("Real-World RL Applications")
        hdr.setStyleSheet("color:#10B981; font-size:13pt; font-weight:900; "
                          "border-bottom:1px solid rgba(16,185,129,0.3); padding-bottom:6px;")
        inner_lay.addWidget(hdr)

        for title, year, desc, result, color in APPLICATIONS:
            card = ApplicationCard(title, year, desc, result, color)
            inner_lay.addWidget(card)

        inner_lay.addStretch()
        scroll.setWidget(inner); lay.addWidget(scroll)
```

- [ ] **Step 3: Commit**
```bash
cd /Users/jsw/20260406/rl0407
git add ui/views/concepts/policy_based_view.py ui/views/concepts/applications_view.py
git commit -m "feat: concept views — policy-based and applications"
```

---

## Task 17: PlaygroundView

**Files:**
- Create: `ui/views/playground_view.py`

- [ ] **Step 1: Write `playground_view.py`**

`ui/views/playground_view.py`:
```python
from __future__ import annotations
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                                QSplitter, QLabel)
from PySide6.QtCore import Qt, Signal

from ui.components.glass_panel import GlassPanel
from ui.components.slider_group import SliderGroup
from ui.components.learning_curve import LearningCurve
from ui.components.status_bar import StatusBar
from ui.visualizations.gridworld_view import GridWorldView
from ui.visualizations.cartpole_view import CartPoleView
from ui.visualizations.maze_view import MazeView
from controllers.training_controller import TrainingController


class GridWorldTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctrl = TrainingController(self)
        self._ctrl.agent_moved.connect(self._on_agent)
        self._ctrl.episode_done.connect(self._on_episode)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(12)

        # Left: env viz
        left = QVBoxLayout()
        self._gw_view = GridWorldView()
        left.addWidget(self._gw_view)
        self._status = StatusBar()
        left.addWidget(self._status)
        lay.addLayout(left, 3)

        # Right: controls + curve
        right = QVBoxLayout(); right.setSpacing(12)
        self._sliders = SliderGroup([
            ("Grid Size", 4, 10, 4),
            ("Alpha α", 1, 20, 2),    # ×0.05
            ("Gamma γ", 50, 99, 95),  # ×0.01
            ("Epsilon ε", 5, 50, 20), # ×0.01
            ("Speed", 1, 50, 10),     # steps/sec
        ])
        panel = GlassPanel(); panel_lay = QVBoxLayout(panel)
        panel_lay.addWidget(QLabel("Hyperparameters"))
        panel_lay.addWidget(self._sliders)
        right.addWidget(panel)

        self._curve = LearningCurve()
        curve_panel = GlassPanel(); cl = QVBoxLayout(curve_panel)
        cl.addWidget(QLabel("Episode Rewards"))
        cl.addWidget(self._curve)
        right.addWidget(curve_panel)

        # Start/Stop buttons
        from PySide6.QtWidgets import QPushButton
        btn_lay = QHBoxLayout()
        self._btn_start = QPushButton("▶  Train")
        self._btn_stop = QPushButton("■  Stop")
        self._btn_reset = QPushButton("↺  Reset")
        for btn in [self._btn_start, self._btn_stop, self._btn_reset]:
            btn.setFixedHeight(36)
            btn_lay.addWidget(btn)
        right.addLayout(btn_lay)
        right.addStretch()
        lay.addLayout(right, 2)

        self._btn_start.clicked.connect(self._start)
        self._btn_stop.clicked.connect(self._ctrl.stop)
        self._btn_reset.clicked.connect(self._reset)
        self._sliders.value_changed.connect(self._on_slider)
        self._setup_env()

    def _setup_env(self):
        vals = self._sliders.values()
        n = vals[0]; alpha = vals[1] * 0.05; gamma = vals[2] * 0.01
        epsilon = vals[3] * 0.01; speed = vals[4]
        self._ctrl.setup(
            "gridworld", {"n": n},
            "qlearning", {"alpha": alpha, "gamma": gamma, "epsilon": epsilon}
        )
        self._ctrl.set_speed(speed)
        env = self._ctrl.env
        if env: self._gw_view.load_env(env)

    def _start(self):
        self._setup_env(); self._ctrl.start()

    def _reset(self):
        self._ctrl.reset(); self._curve.clear()
        self._status.set_text("Reset — press Train to start")

    def _on_agent(self, state: int, reward: float):
        self._gw_view.update_agent(state, reward)
        q = self._ctrl.get_q_table()
        if q is not None: self._gw_view.update_q(q)

    def _on_episode(self, ep: int, total_r: float):
        self._curve.add_point(total_r)
        self._status.set_text(f"Episode {ep}  |  Reward: {total_r:.2f}")

    def _on_slider(self, idx: int, val: int):
        if idx == 4:
            self._ctrl.set_speed(val)

    def set_phase(self, phase: float):
        self._gw_view.set_phase(phase)


class CartPoleTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctrl = TrainingController(self)
        self._ctrl.cartpole_state.connect(self._on_cp_state)
        self._ctrl.episode_done.connect(self._on_episode)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(12)

        left = QVBoxLayout()
        self._cp_view = CartPoleView()
        left.addWidget(self._cp_view)
        self._status = StatusBar()
        left.addWidget(self._status)
        lay.addLayout(left, 3)

        right = QVBoxLayout(); right.setSpacing(12)
        self._sliders = SliderGroup([
            ("Alpha α", 1, 20, 2),
            ("Gamma γ", 50, 99, 95),
            ("Epsilon ε", 5, 50, 20),
            ("Speed", 1, 50, 10),
        ])
        panel = GlassPanel(); panel_lay = QVBoxLayout(panel)
        panel_lay.addWidget(QLabel("Hyperparameters"))
        panel_lay.addWidget(self._sliders)
        right.addWidget(panel)

        self._curve = LearningCurve()
        cp = GlassPanel(); cl = QVBoxLayout(cp)
        cl.addWidget(QLabel("Episode Rewards")); cl.addWidget(self._curve)
        right.addWidget(cp)

        from PySide6.QtWidgets import QPushButton
        btn_lay = QHBoxLayout()
        self._btn_start = QPushButton("▶  Train")
        self._btn_stop = QPushButton("■  Stop")
        self._btn_reset = QPushButton("↺  Reset")
        for btn in [self._btn_start, self._btn_stop, self._btn_reset]:
            btn.setFixedHeight(36); btn_lay.addWidget(btn)
        right.addLayout(btn_lay); right.addStretch()
        lay.addLayout(right, 2)

        self._btn_start.clicked.connect(self._start)
        self._btn_stop.clicked.connect(self._ctrl.stop)
        self._btn_reset.clicked.connect(self._reset)
        self._setup_env()

    def _setup_env(self):
        vals = self._sliders.values()
        alpha = vals[0] * 0.05; gamma = vals[1] * 0.01
        epsilon = vals[2] * 0.01; speed = vals[3]
        self._ctrl.setup("cartpole", {}, "qlearning",
                         {"alpha": alpha, "gamma": gamma, "epsilon": epsilon})
        self._ctrl.set_speed(speed)

    def _start(self): self._setup_env(); self._ctrl.start()
    def _reset(self): self._ctrl.reset(); self._curve.clear()

    def _on_cp_state(self, x, xd, th, thd, done):
        self._cp_view.update_state(x, xd, th, thd, done)

    def _on_episode(self, ep, total_r):
        self._curve.add_point(total_r)
        self._status.set_text(f"Episode {ep}  |  Reward: {total_r:.2f}")


class MazeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctrl = TrainingController(self)
        self._ctrl.maze_moved.connect(self._on_maze)
        self._ctrl.episode_done.connect(self._on_episode)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12); lay.setSpacing(12)

        left = QVBoxLayout()
        self._maze_view = MazeView()
        left.addWidget(self._maze_view)
        self._status = StatusBar()
        left.addWidget(self._status)
        lay.addLayout(left, 3)

        right = QVBoxLayout(); right.setSpacing(12)
        self._sliders = SliderGroup([
            ("Rows", 3, 12, 5),
            ("Cols", 3, 12, 5),
            ("Alpha α", 1, 20, 2),
            ("Gamma γ", 50, 99, 95),
            ("Speed", 1, 50, 10),
        ])
        panel = GlassPanel(); pl = QVBoxLayout(panel)
        pl.addWidget(QLabel("Hyperparameters")); pl.addWidget(self._sliders)
        right.addWidget(panel)

        self._curve = LearningCurve()
        cp = GlassPanel(); cl = QVBoxLayout(cp)
        cl.addWidget(QLabel("Episode Rewards")); cl.addWidget(self._curve)
        right.addWidget(cp)

        from PySide6.QtWidgets import QPushButton
        btn_lay = QHBoxLayout()
        self._btn_start = QPushButton("▶  Train")
        self._btn_stop = QPushButton("■  Stop")
        self._btn_reset = QPushButton("↺  Reset")
        for btn in [self._btn_start, self._btn_stop, self._btn_reset]:
            btn.setFixedHeight(36); btn_lay.addWidget(btn)
        right.addLayout(btn_lay); right.addStretch()
        lay.addLayout(right, 2)

        self._btn_start.clicked.connect(self._start)
        self._btn_stop.clicked.connect(self._ctrl.stop)
        self._btn_reset.clicked.connect(self._reset)
        self._setup_env()

    def _setup_env(self):
        vals = self._sliders.values()
        rows = vals[0]; cols = vals[1]
        alpha = vals[2] * 0.05; gamma = vals[3] * 0.01; speed = vals[4]
        self._ctrl.setup("maze", {"rows": rows, "cols": cols},
                         "qlearning", {"alpha": alpha, "gamma": gamma, "epsilon": 0.2})
        self._ctrl.set_speed(speed)
        env = self._ctrl.env
        if env: self._maze_view.load_maze(env)

    def _start(self): self._setup_env(); self._ctrl.start()
    def _reset(self): self._ctrl.reset(); self._curve.clear()

    def _on_maze(self, pos: tuple, reward: float):
        self._maze_view.update_agent(pos)

    def _on_episode(self, ep, total_r):
        self._curve.add_point(total_r)
        self._status.set_text(f"Episode {ep}  |  Reward: {total_r:.2f}")


class PlaygroundView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabBar::tab { color: rgba(255,255,255,0.7); padding: 8px 20px; "
            "background: transparent; border: none; }"
            "QTabBar::tab:selected { color: #00D4FF; "
            "border-bottom: 2px solid #00D4FF; }"
        )
        self._gw_tab = GridWorldTab()
        self._cp_tab = CartPoleTab()
        self._maze_tab = MazeTab()
        self._tabs.addTab(self._gw_tab, "GridWorld")
        self._tabs.addTab(self._cp_tab, "CartPole")
        self._tabs.addTab(self._maze_tab, "Maze")
        lay.addWidget(self._tabs)

    def set_phase(self, phase: float):
        self._gw_tab.set_phase(phase)
```

- [ ] **Step 2: Commit**
```bash
cd /Users/jsw/20260406/rl0407
git add ui/views/playground_view.py
git commit -m "feat: PlaygroundView — GridWorld, CartPole, Maze training tabs"
```

---

## Task 18: ArenaView

**Files:**
- Create: `ui/views/arena_view.py`

- [ ] **Step 1: Write `arena_view.py`**

`ui/views/arena_view.py`:
```python
from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton)
from PySide6.QtCore import Qt

from ui.components.glass_panel import GlassPanel
from ui.components.learning_curve import LearningCurve
from ui.components.status_bar import StatusBar
from ui.visualizations.gridworld_view import GridWorldView
from ui.views.concepts.policy_based_view import PolicyGradientWidget
from controllers.arena_controller import ArenaController
from ui.theme import CYAN, VIOLET


class ArenaView(QWidget):
    """Side-by-side Q-Learning vs REINFORCE head-to-head."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctrl = ArenaController(self)
        self._ctrl.step_done.connect(self._on_step)
        self._ctrl.q_table_updated.connect(self._on_q)
        self._ctrl.rf_policy_updated.connect(self._on_rf)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16); lay.setSpacing(12)

        # Header
        hdr = QLabel("Arena — Q-Learning vs REINFORCE")
        hdr.setStyleSheet("color: #FFFFFF; font-size: 14pt; font-weight: 900;")
        lay.addWidget(hdr)

        # Top: two GridWorld views
        env_row = QHBoxLayout(); env_row.setSpacing(16)

        ql_panel = GlassPanel(); ql_lay = QVBoxLayout(ql_panel)
        ql_title = QLabel("Q-Learning")
        ql_title.setStyleSheet(f"color: #00D4FF; font-size:11pt; font-weight:700;")
        ql_lay.addWidget(ql_title)
        self._ql_view = GridWorldView()
        ql_lay.addWidget(self._ql_view)
        env_row.addWidget(ql_panel)

        rf_panel = GlassPanel(); rf_lay = QVBoxLayout(rf_panel)
        rf_title = QLabel("REINFORCE")
        rf_title.setStyleSheet(f"color: #7C3AED; font-size:11pt; font-weight:700;")
        rf_lay.addWidget(rf_title)
        self._rf_view = GridWorldView()
        rf_lay.addWidget(self._rf_view)
        env_row.addWidget(rf_panel)

        lay.addLayout(env_row, 3)

        # Load initial envs
        self._ql_view.load_env(self._ctrl._env_ql)
        self._rf_view.load_env(self._ctrl._env_rf)

        # Bottom: two learning curves
        curve_row = QHBoxLayout(); curve_row.setSpacing(16)

        ql_curve_panel = GlassPanel(); qcl = QVBoxLayout(ql_curve_panel)
        qcl.addWidget(QLabel("Q-Learning Returns"))
        self._ql_curve = LearningCurve(line_color=CYAN)
        qcl.addWidget(self._ql_curve)
        curve_row.addWidget(ql_curve_panel)

        rf_curve_panel = GlassPanel(); rcl = QVBoxLayout(rf_curve_panel)
        rcl.addWidget(QLabel("REINFORCE Returns"))
        self._rf_curve = LearningCurve(line_color=VIOLET)
        rcl.addWidget(self._rf_curve)
        curve_row.addWidget(rf_curve_panel)

        lay.addLayout(curve_row, 2)

        # Controls
        btn_row = QHBoxLayout()
        self._btn_start = QPushButton("▶  Start Arena")
        self._btn_stop = QPushButton("■  Stop")
        self._btn_reset = QPushButton("↺  Reset")
        for btn in [self._btn_start, self._btn_stop, self._btn_reset]:
            btn.setFixedHeight(40); btn_row.addWidget(btn)
        lay.addLayout(btn_row)

        self._status = StatusBar()
        lay.addWidget(self._status)

        self._btn_start.clicked.connect(self._ctrl.start)
        self._btn_stop.clicked.connect(self._ctrl.stop)
        self._btn_reset.clicked.connect(self._reset)

    def _on_step(self, ep: int, ql_r: float, rf_r: float):
        self._ql_curve.add_point(ql_r)
        self._rf_curve.add_point(rf_r)
        winner = "Q-Learning" if ql_r > rf_r else "REINFORCE" if rf_r > ql_r else "Tie"
        self._status.set_text(
            f"Ep {ep}  |  Q-Learning: {ql_r:.2f}  |  REINFORCE: {rf_r:.2f}  |  → {winner}")

    def _on_q(self, q_table: np.ndarray):
        self._ql_view.update_q(q_table)

    def _on_rf(self, policy_w: np.ndarray):
        # Convert policy weights to soft Q-table for heatmap visualization
        n_states = policy_w.shape[0]; n_actions = policy_w.shape[1]
        # Use raw weights as proxy Q-values for heatmap display
        self._rf_view.update_q(policy_w)

    def _reset(self):
        self._ctrl.reset()
        self._ql_curve.clear(); self._rf_curve.clear()
        self._ql_view.load_env(self._ctrl._env_ql)
        self._rf_view.load_env(self._ctrl._env_rf)
        self._status.set_text("Reset — press Start Arena")
```

- [ ] **Step 2: Commit**
```bash
cd /Users/jsw/20260406/rl0407
git add ui/views/arena_view.py
git commit -m "feat: ArenaView — Q-Learning vs REINFORCE head-to-head comparison"
```

---

## Task 19: ConceptsView Router

**Files:**
- Create: `ui/views/concepts_view.py`

- [ ] **Step 1: Write `concepts_view.py`**

`ui/views/concepts_view.py`:
```python
from __future__ import annotations
from PySide6.QtWidgets import QWidget, QStackedWidget, QVBoxLayout

from ui.views.concepts.rl_basics_view import RLBasicsView
from ui.views.concepts.value_based_view import ValueBasedView
from ui.views.concepts.policy_based_view import PolicyBasedView
from ui.views.concepts.applications_view import ApplicationsView


class ConceptsView(QWidget):
    """Routes sub-pages: rl-basics, value-based, policy-based, applications."""

    PAGE_MAP = {
        "concepts/rl-basics": 0,
        "concepts/value-based": 1,
        "concepts/policy-based": 2,
        "concepts/applications": 3,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        self._rl_basics = RLBasicsView()
        self._value = ValueBasedView()
        self._policy = PolicyBasedView()
        self._apps = ApplicationsView()

        self._stack.addWidget(self._rl_basics)
        self._stack.addWidget(self._value)
        self._stack.addWidget(self._policy)
        self._stack.addWidget(self._apps)
        lay.addWidget(self._stack)

    def show_page(self, page_id: str):
        idx = self.PAGE_MAP.get(page_id, 0)
        self._stack.setCurrentIndex(idx)

    def set_phase(self, phase: float):
        self._rl_basics.set_phase(phase)
        self._value.set_phase(phase)
        self._policy.set_phase(phase)

    def update_q(self, q: "np.ndarray"):
        self._value.update_q(q)

    def update_policy_probs(self, probs: "np.ndarray"):
        self._policy.update_probs(probs)

    def add_reinforce_reward(self, r: float):
        self._policy.add_reward(r)
```

- [ ] **Step 2: Commit**
```bash
cd /Users/jsw/20260406/rl0407
git add ui/views/concepts_view.py
git commit -m "feat: ConceptsView — stacked router for all concept sub-pages"
```

---

## Task 20: MainWindow

**Files:**
- Create: `ui/main_window.py`

- [ ] **Step 1: Write `main_window.py`**

`ui/main_window.py`:
```python
from __future__ import annotations
from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QStackedWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor

from ui.theme import apply_theme, BG
from ui.components.sidebar import Sidebar
from ui.views.concepts_view import ConceptsView
from ui.views.playground_view import PlaygroundView
from ui.views.arena_view import ArenaView
from controllers.concept_controller import ConceptController


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL AI Dashboard")
        self.resize(1280, 820)
        self.setMinimumSize(960, 640)
        apply_theme(self)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        self._sidebar = Sidebar()
        self._sidebar.page_requested.connect(self._navigate)
        root.addWidget(self._sidebar)

        # Content stack
        self._stack = QStackedWidget()
        root.addWidget(self._stack, 1)

        self._concepts_view = ConceptsView()
        self._playground_view = PlaygroundView()
        self._arena_view = ArenaView()

        self._stack.addWidget(self._concepts_view)   # 0
        self._stack.addWidget(self._playground_view)  # 1
        self._stack.addWidget(self._arena_view)       # 2

        # Global phase animation (sidebar + concept widgets)
        self._concept_ctrl = ConceptController(self)
        self._concept_ctrl.phase_updated.connect(self._on_phase)
        self._concept_ctrl.start(speed=0.008)

        self._navigate("concepts/rl-basics")

    # ── navigation ───────────────────────────────────────────────────────────
    def _navigate(self, page_id: str):
        self._sidebar.set_active(page_id)
        if page_id.startswith("concepts/"):
            self._stack.setCurrentIndex(0)
            self._concepts_view.show_page(page_id)
        elif page_id == "playground":
            self._stack.setCurrentIndex(1)
        elif page_id == "arena":
            self._stack.setCurrentIndex(2)

    def _on_phase(self, phase: float):
        self._sidebar.set_phase(phase)
        self._concepts_view.set_phase(phase)
        self._playground_view.set_phase(phase)
```

- [ ] **Step 2: Commit**
```bash
cd /Users/jsw/20260406/rl0407
git add ui/main_window.py
git commit -m "feat: MainWindow — sidebar navigation, stacked views, phase animation"
```

---

## Task 21: main.py + Smoke Test

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write `main.py`**

`main.py`:
```python
import sys
import os

# Ensure repo root on path
sys.path.insert(0, os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RL AI Dashboard")
    app.setApplicationVersion("1.0.0")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke test**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -c "
import sys; sys.path.insert(0, '.')
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv)
from ui.main_window import MainWindow
win = MainWindow()
win.show()
print(f'Window: {win.isVisible()} {win.width()} x {win.height()}')
print(f'Stack pages: {win._stack.count()}')
print(f'Smoke test PASS')
from PySide6.QtCore import QTimer; QTimer.singleShot(1000, app.quit); app.exec()
"
```

Expected:
```
Window: True 1280 x 820
Stack pages: 3
Smoke test PASS
```

- [ ] **Step 3: Commit**
```bash
cd /Users/jsw/20260406/rl0407
git add main.py
git commit -m "feat: main.py entrypoint + smoke test"
```

---

## Task 22: Run All Tests

**Files:** None (test execution only)

- [ ] **Step 1: Run environment tests**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/test_environments.py -v
```
Expected: All PASS

- [ ] **Step 2: Run algorithm tests**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/test_algorithms.py -v
```
Expected: All PASS

- [ ] **Step 3: Run replay buffer tests**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/test_replay_buffer.py -v
```
Expected: All PASS

- [ ] **Step 4: Run full test suite**
```bash
cd /Users/jsw/20260406/rl0407 && python3 -m pytest tests/ -v
```
Expected: All PASS

- [ ] **Step 5: Final commit**
```bash
cd /Users/jsw/20260406/rl0407
git add .
git commit -m "test: all tests passing — RL Dashboard complete"
```