# Software Design Description
## RL AI Dashboard — Technical Architecture
**Document ID:** SDD-RLAI-001  
**Version:** 1.0  
**Date:** 2026-04-07  
**Standard:** IEEE Std 1016-2009  
**Parent Document:** SRS-RLAI-001  
**Status:** Approved  

---

## Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-04-07 | JSW | Initial architecture — full system design |

---

## Table of Contents

1. Introduction  
2. System Architecture Overview  
3. Component Design  
4. Data Design  
5. Interface Design  
6. Concurrency and Timing Design  
7. Rendering Pipeline  
8. Algorithm Specifications  
9. File Structure  
Appendix A: Signal/Slot Registry  
Appendix B: Color System  
Appendix C: Animation Contract  

---

## 1. Introduction

### 1.1 Purpose

This Software Design Description (SDD) defines the architectural design of **RL AI Dashboard**. It specifies the decomposition into components, inter-component communication via Qt Signal/Slot, the decoupled training/visualization concurrency model, all rendering contracts for QPainter widgets, and the complete specification of all RL algorithms in NumPy.

### 1.2 Design Principles

1. **Zero-Qt Core** — `core/` packages contain no Qt imports. All RL environments and algorithms communicate solely through NumPy arrays and Python callbacks.
2. **Signal/Slot Boundary** — The only legal crossing point between Core and UI is the `Controller` layer, which subscribes to Core callbacks and emits Qt signals.
3. **QTimer Decoupling** — Training steps execute inside `QTimer.timeout` slots at configurable intervals. Visualization updates are driven by a separate 33ms `QTimer` (30 FPS). Neither blocks the other.
4. **QPainter-Only Rendering** — All visualization widgets override `paintEvent(QPainter)`. No Matplotlib, no Qt Charts.
5. **Glassmorphism Token System** — All colors, opacity, glow parameters are defined in `ui/theme.py` as a single source of truth.

### 1.3 Design Viewpoints

| View | Description |
|------|-------------|
| Logical | Package and class decomposition |
| Process | QTimer + main thread concurrency model |
| Physical | Deployment: single process, single thread |
| Development | File structure, module boundaries |
| Scenarios | Training loop, concept page interaction, comparison arena race |

---

## 2. System Architecture Overview

### 2.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Presentation Layer (ui/)                                │
│  MainWindow → Sidebar → ContentStack                    │
│  ├── views/ConceptsView (4 sub-pages)                   │
│  ├── views/PlaygroundView (GridWorld, CartPole, Maze)   │
│  └── views/ArenaView                                    │
│  ├── visualizations/ (all QPainter widgets)             │
│  └── components/ (sliders, panels, charts)              │
├─────────────────────────────────────────────────────────┤
│  Controller Layer (controllers/)                         │
│  Mediates Core↔UI. Emits Qt signals from NumPy data.   │
│  TrainingController, ConceptController, ArenaController │
├─────────────────────────────────────────────────────────┤
│  Core Layer (core/)                                      │
│  ├── environments/ (GridWorld, CartPole, Maze)          │
│  └── algorithms/ (QLearning, REINFORCE, ReplayBuffer)   │
│  Pure NumPy. Zero Qt imports.                           │
└─────────────────────────────────────────────────────────┘
```

Inter-layer rule: Core → Controller (Python callback), Controller → UI (Qt Signal), UI → Controller (Qt Slot). UI never calls Core directly.

### 2.2 Main Process Thread Model

```
Main Thread (Qt Event Loop)
    │
    ├── render_timer: QTimer(33ms) ─────→ all QPainter widgets.update()
    │
    ├── train_timer: QTimer(variable) ──→ TrainingController._step()
    │       │
    │       └── calls QLearning.step() / REINFORCE.episode() [NumPy, fast]
    │               │
    │               └── emits: step_done(StepData), episode_done(EpisodeData)
    │
    └── concept_timer: QTimer(33ms) ───→ ConceptAnimations.tick()
```

**Critical design decision:** RL training runs on the main thread inside a `QTimer` callback. This is intentional:
- Eliminates thread-safety concerns for shared NumPy arrays
- PySide6's `QTimer` callbacks are non-blocking (they return before the next event)
- Training speed is controlled by the number of steps per callback invocation (1/10/100/∞)
- At "∞ speed", the timer fires as fast as the event loop allows; visualization updates are gated separately

### 2.3 Package Dependency Graph

```
main.py
└── ui/main_window.py
    ├── ui/views/*.py
    │   └── ui/visualizations/*.py
    │       └── ui/components/*.py
    │           └── ui/theme.py
    └── controllers/*.py
        └── core/environments/*.py
        └── core/algorithms/*.py
```

No circular imports permitted.

---

## 3. Component Design

### 3.1 Core Layer

#### 3.1.1 `core/environments/gridworld.py` — `GridWorld`

**Responsibility:** N×N MDP grid environment. Pure NumPy.

```python
class GridWorld:
    ACTIONS = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}  # Up/Down/Left/Right
    
    def __init__(self, n: int = 6, obstacles: list = None, seed: int = 42)
    def reset(self) -> int                    # returns state index
    def step(self, action: int) -> tuple      # (next_state, reward, done)
    def state_to_rc(self, state: int) -> tuple  # (row, col)
    def rc_to_state(self, row, col) -> int
    
    # State: int in [0, n*n)
    # Reward: +1.0 goal, -10.0 cliff, -0.01 step cost
    # Terminal: goal reached or cliff
```

**Key properties:**
- `n_states = n * n`
- `n_actions = 4`
- `start_state = 0` (top-left)
- `goal_state = n*n - 1` (bottom-right)
- `obstacles: list[int]` — impassable cells

#### 3.1.2 `core/environments/cartpole.py` — `CartPole`

**Responsibility:** Newtonian CartPole physics. Pure NumPy.

```python
class CartPole:
    # Physics constants
    GRAVITY = 9.8; MASSCART = 1.0; MASSPOLE = 0.1
    LENGTH = 0.5; FORCE_MAG = 10.0; TAU = 0.02  # seconds per step
    
    def __init__(self)
    def reset(self) -> np.ndarray          # state shape (4,): [x, x_dot, theta, theta_dot]
    def step(self, action: int) -> tuple   # (next_state, reward, done)
    def discretize(self, state, bins) -> int  # state → discrete index for Q-table
    
    # Equations of motion (Euler integration):
    # F = force ∈ {-FORCE_MAG, +FORCE_MAG}
    # total_mass = MASSCART + MASSPOLE
    # polemass_length = MASSPOLE * LENGTH
    # temp = (F + polemass_length * theta_dot² * sin(theta)) / total_mass
    # thetaacc = (GRAVITY*sin(theta) - cos(theta)*temp) / 
    #            (LENGTH*(4/3 - MASSPOLE*cos(theta)²/total_mass))
    # xacc = temp - polemass_length*thetaacc*cos(theta)/total_mass
    # Euler: x += TAU*x_dot, x_dot += TAU*xacc, theta += TAU*theta_dot, theta_dot += TAU*thetaacc
    # Terminal: |x| > 2.4 or |theta| > 12° (0.2094 rad)
```

#### 3.1.3 `core/environments/maze.py` — `Maze`

**Responsibility:** Perfect maze (Recursive Backtracking), BFS solution, Q-Learning interface.

```python
class Maze:
    def __init__(self, width: int = 15, height: int = 15, seed: int = None)
    def _generate(self)          # Recursive Backtracking DFS
    def reset(self) -> int       # returns start state
    def step(self, action: int) -> tuple  # (next_state, reward, done)
    def bfs_shortest(self) -> int  # optimal path length
    def get_walls(self) -> np.ndarray  # shape (H, W, 4) bool: [N,S,E,W] walls
    
    # n_states = width * height; n_actions = 4
    # Reward: +10 goal, -0.1 step, -1.0 wall hit (stays in place)
```

#### 3.1.4 `core/algorithms/qlearning.py` — `QLearning`

**Responsibility:** Tabular Q-Learning with ε-greedy exploration.

```python
class QLearning:
    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01)
    
    def select_action(self, state: int) -> int
    # Returns argmax Q[state] with prob (1-ε), random with prob ε
    
    def update(self, state, action, reward, next_state, done)
    # Q[s,a] += α * (r + γ * max(Q[s']) * (1-done) - Q[s,a])
    
    def run_episode(self, env) -> tuple  # (total_reward, steps, trajectory)
    # trajectory: list of (state, action, reward) tuples
    
    def get_policy(self) -> np.ndarray   # shape (n_states,) — argmax per state
    def get_q_values(self) -> np.ndarray  # shape (n_states, n_actions)
    
    # Properties
    q_table: np.ndarray  # shape (n_states, n_actions), initialized to 0
    epsilon: float       # decays each episode
```

#### 3.1.5 `core/algorithms/reinforce.py` — `REINFORCE`

**Responsibility:** Monte Carlo policy gradient with softmax linear policy.

```python
class LinearPolicy:
    """Softmax linear policy: π(a|s) = softmax(Ws + b)"""
    def __init__(self, n_states: int, n_actions: int)
    
    def forward(self, state: int) -> np.ndarray  # (n_actions,) probabilities
    # One-hot encode state, compute Ws+b, apply softmax
    
    def softmax(self, x: np.ndarray) -> np.ndarray
    def log_softmax(self, x: np.ndarray) -> np.ndarray
    
    W: np.ndarray  # shape (n_actions, n_states)
    b: np.ndarray  # shape (n_actions,)


class REINFORCE:
    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.001, gamma: float = 0.99)
    
    def select_action(self, state: int) -> tuple  # (action, log_prob)
    
    def compute_returns(self, rewards: list) -> np.ndarray
    # G[t] = sum_{k=t}^{T} γ^(k-t) r[k]   (backward sum)
    
    def update(self, states, actions, log_probs, returns)
    # Normalize returns: G = (G - G.mean()) / (G.std() + 1e-8)
    # For each t: gradient = -log_prob[t] * G[t]
    # W -= alpha * gradient (manual backprop through linear+softmax)
    
    def run_episode(self, env) -> tuple  # (total_reward, steps, trajectory)
    
    def get_policy_probs(self) -> np.ndarray  # shape (n_states, n_actions)
```

#### 3.1.6 `core/algorithms/replay_buffer.py` — `ReplayBuffer`

**Responsibility:** Fixed-size circular experience replay buffer for DQN concept visualization.

```python
class ReplayBuffer:
    def __init__(self, capacity: int = 1000)
    def push(self, state, action, reward, next_state, done)
    def sample(self, batch_size: int) -> tuple  # (states, actions, rewards, next_states, dones)
    def __len__(self) -> int
    
    # Internal: circular np.ndarray storage for states, actions, rewards, next_states, dones
    # Fields for visualization: current_ptr (write head), current_size
```

### 3.2 Controller Layer

#### 3.2.1 `controllers/training_controller.py` — `TrainingController`

**Responsibility:** Bridges Core algorithms and UI. Owns QTimers. Emits training signals.

```python
class TrainingController(QObject):
    # Signals
    step_done  = Signal(dict)     # {state, action, reward, next_state, done, epsilon}
    episode_done = Signal(dict)   # {episode, total_reward, steps, trajectory}
    q_updated  = Signal(np.ndarray, np.ndarray)  # q_table, policy_arr
    policy_updated = Signal(np.ndarray)  # policy_probs (n_states, n_actions)
    
    def __init__(self, env, algorithm)
    def set_speed(self, mode: str)  # 'step', 'normal', 'fast', 'max'
    def start(self); def stop(self); def reset(self)
    
    def _on_train_tick(self)  # QTimer slot — runs N steps per call
```

#### 3.2.2 `controllers/concept_controller.py` — `ConceptController`

**Responsibility:** Drives concept page animations. Owns the 33ms concept QTimer.

```python
class ConceptController(QObject):
    tick = Signal(float)  # normalized time [0, 1] cycling at 2Hz for animations
    
    def __init__(self)
    def start(self); def stop(self)
```

#### 3.2.3 `controllers/arena_controller.py` — `ArenaController`

**Responsibility:** Manages two TrainingControllers (Q-Learning + REINFORCE) on shared GridWorld.

```python
class ArenaController(QObject):
    both_episode_done = Signal(int, dict, dict)  # episode, ql_stats, rf_stats
    
    def __init__(self)
    def start_race(self); def pause(self); def reset(self)
```

### 3.3 UI Layer

#### 3.3.1 `ui/main_window.py` — `MainWindow`

```python
class MainWindow(QMainWindow):
    # Layout: QHBoxLayout → [Sidebar(200px fixed) | ContentStack(QStackedWidget)]
    # QStackedWidget pages: ConceptsView, PlaygroundView, ArenaView
    # Owns: ConceptController, TrainingController (one active at a time)
    # Keyboard shortcuts: Space, R, 1, 2, 3, C
```

#### 3.3.2 `ui/components/sidebar.py` — `Sidebar`

```python
class Sidebar(QWidget):
    # Custom QPainter: dark glass panel background, neon accent for selected item
    # Items: expandable QTreeWidget-style, but custom-drawn
    page_requested = Signal(str)  # 'concepts/rl-basics', 'playground/gridworld', etc.
```

#### 3.3.3 `ui/components/glass_panel.py` — `GlassPanel`

```python
class GlassPanel(QWidget):
    """Base for all glassmorphism panels. Draws background in paintEvent."""
    # paintEvent: fill rect with QColor(255,255,255,20), draw 1px border QColor(255,255,255,40)
    # Drop shadow via parent widget margin
```

#### 3.3.4 `ui/components/slider_group.py` — `SliderGroup`

```python
class SliderGroup(GlassPanel):
    """Label + QSlider + value display, neon-styled."""
    value_changed = Signal(str, float)  # (param_name, value)
```

#### 3.3.5 `ui/components/learning_curve.py` — `LearningCurveWidget`

```python
class LearningCurveWidget(QWidget):
    """QPainter line chart: episode rewards + 50-ep moving average."""
    def add_episode(self, reward: float)
    def clear(self)
    # paintEvent draws: axes, reward line (semi-transparent), MA line (solid neon)
    # Auto-scales Y axis. Grid lines at 25% intervals.
```

#### 3.3.6 `ui/visualizations/gridworld_view.py` — `GridWorldView`

```python
class GridWorldView(QWidget):
    """Full QPainter GridWorld renderer. Accepts Q-values and policy from controller."""
    def set_data(self, q_table: np.ndarray, policy: np.ndarray,
                 agent_state: int, step_rewards: list)
    # paintEvent:
    #   1. Draw cell backgrounds (heatmap: cyan→magenta based on max Q)
    #   2. Draw policy arrows (direction = argmax Q, opacity = Q confidence)
    #   3. Draw special cells (S=emerald, G=gold, obstacles=dark, cliff=red)
    #   4. Draw reward float-ups (animate Y offset + alpha over 30 frames)
    #   5. Draw agent (layered radial gradient glow + solid core)
    #   6. Draw agent movement interpolation (lerp between current and target cell)
```

#### 3.3.7 `ui/visualizations/cartpole_view.py` — `CartPoleView`

```python
class CartPoleView(QWidget):
    """QPainter CartPole renderer."""
    def set_state(self, state: np.ndarray)  # [x, x_dot, theta, theta_dot]
    # paintEvent:
    #   Track: horizontal neon line spanning widget
    #   Cart: rounded rectangle, position mapped from [-2.4, 2.4] to widget width
    #   Pole: thick line from cart center, angle = state[2]
    #   State bars: 4 horizontal mini-bars showing current values
```

#### 3.3.8 `ui/visualizations/maze_view.py` — `MazeView`

```python
class MazeView(QWidget):
    """QPainter Maze renderer."""
    def set_data(self, maze: Maze, agent_pos: int, path: list, q_table: np.ndarray)
    # paintEvent: walls, corridors, Q-heatmap on cells, path trail, pulsing goal star, agent glow
```

#### 3.3.9 `ui/views/concepts/` — Concept Sub-Pages

Each concept sub-page is a `QWidget` with a `QScrollArea` layout containing `GlassPanel` containers.

| File | Widgets |
|------|---------|
| `rl_basics_view.py` | `AgentEnvLoopWidget`, `MDPGraphWidget`, `MarkovPropertyWidget`, `DiscountFactorWidget` |
| `value_based_view.py` | `QTableWidget`, `BellmanWidget`, `EpsilonGreedyWidget`, `TDUpdateWidget`, `DQNArchWidget`, `ReplayBufferWidget`, `TargetNetWidget` |
| `policy_based_view.py` | `PolicyWidget`, `PolicyGradientWidget`, `REINFORCETrajWidget`, `PolicyUpdateWidget`, `ActionSpaceWidget` |
| `applications_view.py` | `AtariPanel`, `AlphaGoPanel`, `RoboticsPanel` |

---

## 4. Data Design

### 4.1 Training State Data Structures

```python
@dataclass
class StepData:
    state: int
    action: int
    reward: float
    next_state: int
    done: bool
    epsilon: float
    q_values_before: np.ndarray  # shape (n_actions,) at current state
    td_error: float

@dataclass
class EpisodeData:
    episode: int
    total_reward: float
    steps: int
    trajectory: list[tuple]  # [(state, action, reward), ...]
    final_epsilon: float
```

### 4.2 Visualization State

```python
@dataclass
class GridWorldRenderState:
    q_table: np.ndarray      # (n_states, n_actions)
    policy: np.ndarray       # (n_states,) — argmax per state
    agent_state: int
    agent_lerp: float        # 0.0..1.0 interpolation toward next cell
    reward_floats: list      # [(cell, reward, age_frames)]
    episode: int
    total_reward: float
```

### 4.3 Concept Widget State

All concept widgets maintain internal animated state ticked by `ConceptController.tick` signal:

```python
# Example: AgentEnvLoopWidget
class AgentEnvLoopWidget(QWidget):
    _phase: float = 0.0    # 0.0..1.0, cycles every 2 seconds
    _pulse: float = 0.0    # element pulse phase
```

---

## 5. Interface Design

### 5.1 Core → Controller Interface

All Core classes expose callbacks, not signals:

```python
# QLearning / REINFORCE
env.on_step(callback: Callable[[StepData], None])
env.on_episode(callback: Callable[[EpisodeData], None])
```

Controller registers callbacks and emits Qt signals:

```python
def _on_env_step(self, data: StepData):
    self.step_done.emit(data.__dict__)
```

### 5.2 Controller → View Interface

Views connect to controller signals in `PlaygroundView.__init__`:

```python
ctrl.q_updated.connect(gridworld_view.set_data)
ctrl.episode_done.connect(learning_curve.add_episode)
ctrl.step_done.connect(self._update_status)
```

### 5.3 Hyperparameter Slider → Controller Interface

```python
slider_group.value_changed.connect(self._on_param_changed)

def _on_param_changed(self, name: str, value: float):
    if name == 'alpha':   self.controller.algorithm.alpha = value
    elif name == 'gamma': self.controller.algorithm.gamma = value
    elif name == 'epsilon': self.controller.algorithm.epsilon = value
```

---

## 6. Concurrency and Timing Design

### 6.1 Timer Configuration

| Timer | Interval | Owner | Purpose |
|-------|----------|-------|---------|
| `render_timer` | 33ms (30 FPS) | `MainWindow` | Triggers `update()` on all visible QPainter widgets |
| `train_timer` | Variable | `TrainingController` | Executes N training steps per tick |
| `concept_timer` | 33ms | `ConceptController` | Drives concept page animations |

### 6.2 Training Speed Modes

| Mode | Steps per tick | Timer interval | Visualization update |
|------|---------------|----------------|---------------------|
| Step | 1 | on-demand (manual) | Every step |
| Normal | 1 | 33ms | Every step |
| Fast | 10 | 16ms | Every 10 steps |
| Max | 100 | 1ms | Every 100 steps (chart only) |

### 6.3 Render/Train Decoupling

The `GridWorldRenderState` is owned by the View. The Controller updates it via signal emission. The 33ms render timer reads the latest state without blocking:

```
train_timer tick (variable) ──→ QLearning.step() ──→ emit q_updated(q_table, policy)
                                                         │
render_timer tick (33ms) ──────────────────────────────→ GridWorldView.update()
                                                         └── reads latest q_table
```

No mutex required: signal delivery is queued into the event loop.

---

## 7. Rendering Pipeline

### 7.1 QPainter Layer Order (GridWorldView)

```
Layer 0: Cell background fill (heatmap gradient)
Layer 1: Cell border (1px semi-transparent white)
Layer 2: Special cell overlays (S/G/obstacle/cliff markers)
Layer 3: Policy arrows (QPainterPath, opacity-variable)
Layer 4: Reward float-up text (animated, fading)
Layer 5: Agent outer glow (radial gradient, large radius)
Layer 6: Agent inner glow (radial gradient, medium radius)
Layer 7: Agent core (solid circle)
Layer 8: Agent interpolation (lerp between prev and current cell)
```

### 7.2 Glassmorphism Panel Render

```python
def paintEvent(self, event):
    p = QPainter(self)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    # Background: near-black with glass tint
    p.fillRect(rect, QColor(255, 255, 255, 15))  # 6% white
    
    # Border: subtle white outline
    p.setPen(QPen(QColor(255, 255, 255, 50), 1))
    p.drawRoundedRect(rect.adjusted(0,0,-1,-1), 12, 12)
    
    # Drop shadow: drawn by parent via margin (8px on all sides)
```

### 7.3 Neon Glow Render Contract

All glowing elements (agent, goal star, active sidebar item) use this 3-layer pattern:

```python
def _draw_glow(p: QPainter, center: QPointF, radius: float, color: QColor):
    # Outer glow
    g1 = QRadialGradient(center, radius * 2.5)
    g1.setColorAt(0, QColor(color.red(), color.green(), color.blue(), 100))
    g1.setColorAt(1, QColor(0,0,0,0))
    p.setBrush(g1); p.setPen(Qt.NoPen)
    p.drawEllipse(center, radius*2.5, radius*2.5)
    
    # Inner glow
    g2 = QRadialGradient(center, radius * 1.5)
    g2.setColorAt(0, QColor(color.red(), color.green(), color.blue(), 180))
    g2.setColorAt(1, QColor(0,0,0,0))
    p.setBrush(g2)
    p.drawEllipse(center, radius*1.5, radius*1.5)
    
    # Core
    p.setBrush(QBrush(color)); p.drawEllipse(center, radius, radius)
```

### 7.4 Learning Curve Widget Render

```
Axes: left edge + bottom edge, 1px white-40%
Grid lines: horizontal at 25% intervals, white-10%
Reward series: 1.5px semi-transparent Cyan (#00D4FF, alpha=120)
Moving average: 2px solid Cyan (#00D4FF, alpha=255)
REINFORCE series (in Arena): Magenta #FF006E
```

---

## 8. Algorithm Specifications

### 8.1 Q-Learning

```
Initialize Q(s,a) = 0 for all s, a

For each episode:
  s ← env.reset()
  While not done:
    With prob ε: a ← random action
    Else:        a ← argmax_a' Q(s, a')
    
    s', r, done ← env.step(a)
    
    target = r + γ · max_a' Q(s', a') · (1 - done)
    Q(s, a) ← Q(s, a) + α · (target - Q(s, a))
    
    s ← s'
  
  ε ← max(ε_min, ε · ε_decay)
```

**Complexity:** O(1) per step. GridWorld 6×6: 36 states × 4 actions = 144 floats.

### 8.2 REINFORCE (Monte Carlo Policy Gradient)

```
Initialize θ (W, b of LinearPolicy)

For each episode:
  s ← env.reset()
  trajectory = []
  While not done:
    π = softmax(W·one_hot(s) + b)        # shape (n_actions,)
    a ~ π                                 # sample action
    s', r, done ← env.step(a)
    trajectory.append((s, a, log_π(a), r))
    s ← s'
  
  Compute returns:
    G[T] = 0
    For t = T-1 downto 0:
      G[t] = r[t] + γ · G[t+1]
  
  Normalize: G = (G - mean(G)) / (std(G) + 1e-8)
  
  Update for each t:
    ∇log π(a_t|s_t) = one_hot(a_t) - π(s_t)   [softmax gradient]
    ΔW += α · G[t] · ∇log π ⊗ one_hot(s_t)ᵀ
    Δb += α · G[t] · ∇log π
  
  W += ΔW; b += Δb
```

**Complexity:** O(T · n_states · n_actions) per episode. GridWorld 6×6, T=100: ~144K float ops.

### 8.3 CartPole Discretization for Q-Learning

```python
BINS = [
    np.linspace(-2.4, 2.4, 11),      # x: 10 bins
    np.linspace(-3.0, 3.0, 11),      # x_dot: 10 bins
    np.linspace(-0.2094, 0.2094, 11), # theta: 10 bins
    np.linspace(-3.0, 3.0, 11),      # theta_dot: 10 bins
]
# Total states = 10^4 = 10000
def discretize(state):
    return sum(
        np.digitize(s, b) * (10**i)
        for i, (s, b) in enumerate(zip(state, BINS))
    )
```

### 8.4 REINFORCE Linear Policy for CartPole

```
Input: state (4,)
Hidden: Dense(64, tanh) — implemented as W1@state + b1, tanh
Output: Dense(2, softmax) — W2@hidden + b2, softmax

Manual gradient:
  δout = one_hot(a) - π   (softmax categorical cross-entropy gradient)
  δhid = (W2.T @ δout) * (1 - tanh²(hidden))  (tanh gradient)
  
  ΔW2 = α · G · δout ⊗ hidden.T
  ΔW1 = α · G · δhid ⊗ state.T
```

---

## 9. File Structure

```
rl0407/
├── main.py                          # Entry point
├── requirements.txt                 # PySide6>=6.6, numpy>=1.26
├── core/
│   ├── __init__.py
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── gridworld.py             # GridWorld MDP
│   │   ├── cartpole.py              # CartPole physics
│   │   └── maze.py                  # Maze (Recursive Backtracking)
│   └── algorithms/
│       ├── __init__.py
│       ├── qlearning.py             # Tabular Q-Learning
│       ├── reinforce.py             # REINFORCE + LinearPolicy
│       └── replay_buffer.py         # Experience replay (DQN concepts)
├── controllers/
│   ├── __init__.py
│   ├── training_controller.py       # Core↔UI bridge, owns train_timer
│   ├── concept_controller.py        # Concept animation timer
│   └── arena_controller.py          # Dual-algorithm comparison
├── ui/
│   ├── __init__.py
│   ├── main_window.py               # MainWindow, render_timer, keyboard shortcuts
│   ├── theme.py                     # Color tokens, glow params, QSS
│   ├── components/
│   │   ├── __init__.py
│   │   ├── sidebar.py               # Navigation sidebar
│   │   ├── glass_panel.py           # GlassPanel base widget
│   │   ├── slider_group.py          # Neon slider + label
│   │   ├── learning_curve.py        # QPainter line chart
│   │   └── status_bar.py            # Episode counter + status
│   ├── visualizations/
│   │   ├── __init__.py
│   │   ├── gridworld_view.py        # Q-heatmap + arrows + agent + reward floats
│   │   ├── cartpole_view.py         # CartPole physics renderer + state bars
│   │   ├── maze_view.py             # Maze walls + path + agent
│   │   └── painter_utils.py         # draw_glow(), draw_arrow(), lerp_color()
│   └── views/
│       ├── __init__.py
│       ├── concepts/
│       │   ├── __init__.py
│       │   ├── rl_basics_view.py    # AgentEnvLoop, MDP, Markov, DiscountFactor
│       │   ├── value_based_view.py  # QTable, Bellman, εgreedy, TD, DQN, Replay, TargetNet
│       │   ├── policy_based_view.py # Policy, PolicyGradient, REINFORCE, ActionSpace
│       │   └── applications_view.py # Atari, AlphaGo, Robotics infographics
│       ├── playground_view.py       # Tabbed: GridWorld / CartPole / Maze
│       └── arena_view.py            # Comparison Arena dual-panel
└── tests/
    ├── __init__.py
    ├── test_environments.py         # GridWorld/CartPole/Maze step tests
    ├── test_algorithms.py           # Q-Learning/REINFORCE correctness tests
    └── test_replay_buffer.py        # ReplayBuffer capacity + sample tests
```

---

## Appendix A: Signal/Slot Registry

| Signal | Emitter | Connected To | Payload |
|--------|---------|--------------|---------|
| `q_updated` | TrainingController | GridWorldView.set_data | (q_table, policy) |
| `policy_updated` | TrainingController | GridWorldView.set_data | (policy_probs,) |
| `episode_done` | TrainingController | LearningCurveWidget.add_episode | EpisodeData dict |
| `step_done` | TrainingController | StatusBar.update | StepData dict |
| `tick` | ConceptController | All concept widget._on_tick | float phase |
| `page_requested` | Sidebar | MainWindow._on_page | str page_id |
| `value_changed` | SliderGroup | PlaygroundView._on_param | (name, float) |
| `both_episode_done` | ArenaController | ArenaView.update_comparison | (ep, ql, rf) |

---

## Appendix B: Color System

```python
# ui/theme.py

BACKGROUND     = QColor(10, 14, 26)       # #0A0E1A — near black
SURFACE_1      = QColor(22, 28, 45)       # #161C2D — card background
SURFACE_2      = QColor(30, 38, 58)       # #1E263A — elevated surface
BORDER         = QColor(255, 255, 255, 40) # white 16%
BORDER_GLOW    = QColor(255, 255, 255, 80) # white 31%

CYAN           = QColor(0, 212, 255)       # #00D4FF — Q-Learning, primary accent
MAGENTA        = QColor(255, 0, 110)       # #FF006E — REINFORCE, danger
VIOLET         = QColor(124, 58, 237)      # #7C3AED — policy/DQN
EMERALD        = QColor(16, 185, 129)      # #10B981 — reward/goal
AMBER          = QColor(245, 158, 11)      # #F59E0B — epsilon/warning
WHITE_80       = QColor(255, 255, 255, 204)
WHITE_60       = QColor(255, 255, 255, 153)
WHITE_40       = QColor(255, 255, 255, 102)

# Heatmap gradient: low Q → high Q
HEATMAP_LOW    = QColor(255, 0, 110, 80)   # transparent magenta
HEATMAP_MID    = QColor(22, 28, 45, 200)   # surface
HEATMAP_HIGH   = QColor(0, 212, 255, 80)   # transparent cyan

# QSS Template (applied to QApplication)
QSS = f"""
QMainWindow, QWidget {{ background: #0A0E1A; color: #E2E8F0; font-size: 11pt; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: #161C2D; width: 8px; border-radius: 4px; }}
QScrollBar::handle:vertical {{ background: #1E263A; border-radius: 4px; min-height: 20px; }}
QPushButton {{
    background: rgba(0,212,255,0.12); color: #00D4FF;
    border: 1px solid rgba(0,212,255,0.3); border-radius: 8px;
    padding: 8px 20px; font-weight: 700;
}}
QPushButton:hover {{ background: rgba(0,212,255,0.25); }}
QPushButton#danger {{ background: rgba(255,0,110,0.12); color: #FF006E; border-color: rgba(255,0,110,0.3); }}
QSlider::groove:horizontal {{ background: #1E263A; height: 6px; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: #00D4FF; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: #00D4FF; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; border: 2px solid #0A0E1A; }}
"""
```

---

## Appendix C: Animation Contract

All animated concept widgets implement this interface:

```python
class AnimatedWidget(QWidget):
    def _on_tick(self, phase: float):
        """Called at 30 FPS with phase ∈ [0, 1) cycling at animation frequency."""
        self._phase = phase
        self.update()  # triggers paintEvent
    
    def _ease_in_out(self, t: float) -> float:
        """Quadratic ease-in-out: t ∈ [0, 1] → [0, 1]"""
        return 2*t*t if t < 0.5 else -1 + (4 - 2*t)*t
    
    def _lerp_color(self, c1: QColor, c2: QColor, t: float) -> QColor:
        r = int(c1.red()   + (c2.red()   - c1.red())   * t)
        g = int(c1.green() + (c2.green() - c1.green()) * t)
        b = int(c1.blue()  + (c2.blue()  - c1.blue())  * t)
        a = int(c1.alpha() + (c2.alpha() - c1.alpha()) * t)
        return QColor(r, g, b, a)
```

**Pulse animation pattern** (used in AgentEnvLoop, goal star, sidebar active item):
```python
# 1Hz pulse: bright at phase=0, dim at phase=0.5
brightness = 0.6 + 0.4 * math.cos(2 * math.pi * phase)
color = QColor(0, int(212 * brightness), int(255 * brightness))
```
