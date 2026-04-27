# Software Requirements Specification
## RL AI Dashboard — Reinforcement Learning Interactive Educational Platform
**Document ID:** SRS-RLAI-001  
**Version:** 1.0  
**Date:** 2026-04-07  
**Standard:** IEEE Std 830-1998  
**Status:** Approved  

---

## Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-04-07 | JSW | Initial specification — full scope |

---

## Table of Contents

1. Introduction  
2. Overall Description  
3. External Interface Requirements  
4. System Features  
5. Non-Functional Requirements  
Appendix A: Glossary  
Appendix B: Requirements Traceability Matrix  
Appendix C: Concept Coverage Map  

---

## 1. Introduction

### 1.1 Purpose

This Software Requirements Specification (SRS) defines the functional and non-functional requirements for **RL AI Dashboard**, a PySide6 desktop application that serves as a comprehensive, interactive educational platform for Reinforcement Learning. The system visualizes core RL theory — from Markov Decision Processes through Deep Q-Networks and policy gradient methods — through animated, interactive demonstrations running entirely on CPU using pure NumPy.

This document governs:
- All functional behavior of the three major application views (Concepts, Playground, Comparison Arena)
- All real-time rendering and animation requirements (QPainter, 30 FPS)
- The decoupled training/visualization architecture mandated by the QTimer-based design

### 1.2 Scope

**Product Name:** RL AI Dashboard  
**Version Covered:** 1.0.0  
**Target Directory:** `20260406/rl0407/`

The system shall provide:

1. **Concepts Module** — Four themed concept pages (RL Basics, Value-Based, Policy-Based, Applications) with fully interactive, animated visualizations of every listed RL concept
2. **Playground Module** — Live algorithm experimentation across three environments (GridWorld, CartPole, Maze) with real-time Q-value heatmaps, policy arrows, episode trajectories, and learning curves
3. **Comparison Arena** — Simultaneous side-by-side execution of Q-Learning vs REINFORCE on a shared environment with synchronized learning curve comparison

The system does **not** cover: network connectivity, model persistence to disk, multi-user sessions, GPU acceleration, or PyTorch/TensorFlow integration. All ML algorithms are implemented in pure NumPy.

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|-----------|
| RL | Reinforcement Learning |
| MDP | Markov Decision Process |
| Q-value | Action-value function Q(s,a) |
| TD | Temporal Difference |
| DQN | Deep Q-Network |
| REINFORCE | Monte Carlo policy gradient algorithm (Williams, 1992) |
| ε-greedy | Epsilon-greedy exploration strategy |
| γ | Discount factor (gamma), ∈ [0, 1] |
| α | Learning rate |
| FPS | Frames Per Second |
| QPainter | PySide6 custom 2D rendering API |
| QTimer | PySide6 timer class for event-driven updates |
| FSM | Finite State Machine |
| Glassmorphism | UI design style: frosted-glass panels, blur, transparency, dark backgrounds |

### 1.4 Overview

Section 2 describes the product from the user's perspective. Section 3 covers external interfaces. Section 4 contains numbered functional requirements (FR-*). Section 5 contains non-functional requirements (NFR-*). Appendices provide supporting artifacts.

---

## 2. Overall Description

### 2.1 Product Perspective

RL AI Dashboard is a standalone desktop application targeting students and practitioners who want to build intuition for RL algorithms through direct experimentation, not reading. It extends the PhysicsAI Simulator product family (SRS-PHYSAI-001 through SRS-PHYSAI-004) with the same layered PySide6 architecture, QTimer-decoupled training loop, and QPainter visualization philosophy — but applied to reinforcement learning rather than physics simulation.

The system is self-contained: all environments are custom implementations (GridWorld grid search, CartPole Newtonian physics, Maze graph traversal), all algorithms run in pure NumPy, and all rendering is performed via QPainter without Matplotlib.

### 2.2 Product Functions Summary

| Module | Primary Function |
|--------|-----------------|
| Concepts / RL Basics | Animated Agent-Environment Loop, MDP graph, Markov Property, Discount Factor slider |
| Concepts / Value-Based | Q-Table inspector, Bellman equation step-through, DQN architecture + replay buffer |
| Concepts / Policy-Based | Policy probability bar chart, REINFORCE trajectory visualization |
| Concepts / Applications | Atari / AlphaGo / Robotics infographic panels |
| Playground / GridWorld | Live Q-Learning or REINFORCE training with heatmap + arrows + glow agent |
| Playground / CartPole | CartPole physics + state readout, training with live curve |
| Playground / Maze | Maze pathfinding with Q-Learning, path overlay animation |
| Comparison Arena | Dual-panel Q-Learning vs REINFORCE side-by-side on GridWorld |

### 2.3 User Characteristics

**Primary users:** RL/AI students at the introductory–intermediate level. They understand basic ML but are learning RL for the first time. They expect visual feedback to confirm their intuition and interactive controls to explore parameter effects.

**Secondary users:** Instructors demonstrating RL concepts in lecture. They need reliable, repeatable demos at specified parameter values.

### 2.4 Assumptions and Dependencies

- Python 3.10+, PySide6 ≥ 6.6, NumPy ≥ 1.26. No other Python packages.
- macOS (primary), Linux compatible, Windows with font fallback.
- CPU-only execution; all algorithms optimized to run at acceptable interactive speeds on a single core.
- Display resolution ≥ 1280 × 820.

---

## 3. External Interface Requirements

### 3.1 User Interface

**FR-UI-01:** The application window shall be at minimum 1400 × 900 pixels with a maximizable layout.

**FR-UI-02:** The global visual theme shall implement Glassmorphism: near-black background (#0A0E1A), frosted-glass panel overlays (white 6–12% opacity + backdrop blur simulation via gradient + border), neon accent colors (Cyan #00D4FF, Magenta #FF006E, Violet #7C3AED, Emerald #10B981).

**FR-UI-03:** A persistent left sidebar (width 200px) shall contain navigation items for Concepts (expandable: RL Basics, Value-Based, Policy-Based, Applications), Playground (expandable: GridWorld, CartPole, Maze), and Comparison Arena. Active item shall glow with the accent color.

**FR-UI-04:** All transitions between views shall animate with an 8-frame (267ms at 30 FPS) fade-through effect.

**FR-UI-05:** A top header bar shall display the current module name, a real-time episode counter (during training), and a Theme toggle (not required for v1.0 but the slot shall exist).

### 3.2 Hardware Interfaces

**FR-HW-01:** All rendering shall run on CPU only. No CUDA or Metal acceleration shall be required.

**FR-HW-02:** The system shall function with ≥ 4 GB RAM. Peak RAM usage shall not exceed 512 MB during any operation.

### 3.3 Software Interfaces

**FR-SW-01:** The system shall import only: `PySide6` (Widgets, Core, Gui) and `NumPy`. No other third-party packages are permitted.

**FR-SW-02:** All rendering shall use `QPainter` with `QWidget.paintEvent`. No Matplotlib or Qt Charts.

---

## 4. System Features

### 4.1 MODULE: Concepts — RL Basics

**FR-CONC-RLB-01:** The RL Basics page shall display an **Agent-Environment Loop** diagram rendered in QPainter: a circular animated arrow connecting Agent → Action → Environment → State+Reward → Agent, with each element pulsing at 1 Hz.

**FR-CONC-RLB-02:** The page shall include an **MDP Graph** visualization: states as nodes (circles), actions as directed edges, transition probabilities as edge weights, rewards as color-coded edge labels. Nodes shall be draggable.

**FR-CONC-RLB-03:** A **Markov Property** panel shall display a timeline of states s₀..sₙ. Clicking any sₜ shall highlight only the relevant conditioning information (sₜ alone) versus a non-Markov history chain, with animated arrows showing the difference.

**FR-CONC-RLB-04:** A **Discount Factor γ** slider (range 0.0–1.0, step 0.01) shall drive a live **Return G visualization**: a bar chart showing r₀, γr₁, γ²r₂, ... γᵗrₜ for a fixed reward sequence. Bars shall animate height in real time as γ changes.

**FR-CONC-RLB-05:** A **Return calculator** shall show the cumulative sum G updating as γ changes.

### 4.2 MODULE: Concepts — Value-Based

**FR-CONC-VB-01:** A **Q-Table inspector** shall display a 4×4 grid of states, each cell showing Q(s, a) values for 4 actions (Up/Down/Left/Right) as color-coded mini-bars within the cell. Values shall be scrollable/zoomable.

**FR-CONC-VB-02:** A **Bellman Equation step-through** panel shall show the equation Q(s,a) ← Q(s,a) + α[r + γ·maxₐ'Q(s',a') - Q(s,a)] with each term highlighted as the user clicks "Next Step". Animated brackets shall show which values are selected.

**FR-CONC-VB-03:** A **ε-greedy exploration** visualization shall show a probability dial: ε = current probability of random action, (1-ε) = exploit. An animated "decision spinner" shall demonstrate random vs greedy selection.

**FR-CONC-VB-04:** A **TD Update flow** diagram shall illustrate the target computation with a two-step arrow: current estimate → target → updated value, showing the TD error δ highlighted in a box.

**FR-CONC-VB-05:** A **DQN Architecture panel** shall render a schematic network diagram (input layer = state dims → hidden layers → output layer = Q-values per action), drawn as connected nodes in QPainter with gradient weights coloring.

**FR-CONC-VB-06:** An **Experience Replay buffer** visualization shall show a circular buffer (ring) with colored slots representing (s, a, r, s') tuples. As the user adds samples (manual or auto), slots fill. A "Mini-batch sample" button shall animate randomly selected slots glowing and being extracted to a training mini-batch display.

**FR-CONC-VB-07:** A **Target Network** panel shall show two network schematics (Online, Target) side by side. A sync counter shall count steps and animate a "copy weights" beam at the configured sync interval (τ slider: 50–500 steps).

### 4.3 MODULE: Concepts — Policy-Based

**FR-CONC-PB-01:** A **Policy π(a|s)** visualization shall show a state grid where each cell contains a probability distribution bar chart over actions. Hovering a cell shall expand it to show all action probabilities as animated bars.

**FR-CONC-PB-02:** A **Policy Gradient Theorem** panel shall display the gradient formula ∇J(θ) = 𝔼[Gt · ∇log π(a|s;θ)] with color-coded terms. An animated arrow shall show the update direction on a 2D policy parameter space plot.

**FR-CONC-PB-03:** A **REINFORCE trajectory** visualizer shall show a complete episode path on a mini-GridWorld. Each step shall display (sₜ, aₜ, rₜ). The Return Gₜ shall be computed backward from the terminal state and displayed above each step as an animated overlay.

**FR-CONC-PB-04:** A **Policy update direction** widget shall show a 2D parameter space (θ₀ vs θ₁) with the current policy point and gradient arrow. As the user clicks "Update", the point shall animate along the gradient direction with a trail.

**FR-CONC-PB-05:** A **Discrete vs Continuous action space** comparison panel shall show: left side = 4 discrete action buttons with discrete softmax distribution; right side = continuous Gaussian distribution curve with mean/std sliders.

### 4.4 MODULE: Concepts — Applications

**FR-CONC-APP-01:** An **Atari** infographic panel shall display a stylized game screen illustration (QPainter pixel art of a Pong-like screen), a DQN architecture mini-diagram, and a text summary of the breakthrough metrics.

**FR-CONC-APP-02:** An **AlphaGo** panel shall display a stylized 19×19 Go board fragment (QPainter), MCTS + Policy Network + Value Network icons, and a timeline of key milestones.

**FR-CONC-APP-03:** A **Robotics** panel shall display a stylized robot arm diagram (QPainter line art), a continuous action space spiral visualization, and a policy evolution trail showing the arm improving over time.

### 4.5 MODULE: Playground — GridWorld

**FR-PG-GW-01:** The GridWorld environment shall be a configurable N×N grid (N ∈ {4, 6, 8}, default 6) with: one Start cell (S), one Goal cell (G), configurable obstacle cells, and one cliff cell (penalty reward -10).

**FR-PG-GW-02:** The Q-value heatmap shall color each cell by max_a Q(s,a) using a cyan-to-magenta gradient, updated every visualization frame (33ms).

**FR-PG-GW-03:** Policy arrows shall be rendered in each non-obstacle cell indicating argmax_a Q(s,a), with arrow opacity proportional to confidence (max Q - min Q over actions, normalized).

**FR-PG-GW-04:** The agent shall be rendered as a glowing circle (radius 16px, neon color, radial gradient) at its current cell. Movement shall animate smoothly (8-frame linear interpolation between cells).

**FR-PG-GW-05:** Hyperparameter sliders in the left panel: Learning Rate α (0.01–1.0), Discount γ (0.0–1.0), Epsilon ε (0.01–1.0), Epsilon Decay (0.99–0.9999), Episodes (100–10000).

**FR-PG-GW-06:** Reward signals shall display as colored float-up text (+1.0 in green, -10.0 in red, -0.01 in grey) at the cell where the reward was received, animating upward and fading over 30 frames.

**FR-PG-GW-07:** A real-time **learning curve** panel shall display Episode Reward vs Episode Number as a QPainter line chart with a 50-episode moving average overlay.

**FR-PG-GW-08:** The training loop shall run via QTimer at adjustable speed: 1 step/tick (educational), 10 steps/tick (normal), 100 steps/tick (fast), max (no visualization throttle, only update chart every 100 episodes).

**FR-PG-GW-09:** Algorithm selector: Q-Learning or REINFORCE. On switch, the Q-table or policy network shall reset.

**FR-PG-GW-10:** A **Q-Table view** panel (expandable) shall show a scrollable numerical table of all Q(s,a) values, color-coded by magnitude.

### 4.6 MODULE: Playground — CartPole

**FR-PG-CP-01:** CartPole physics shall implement: cart mass 1.0 kg, pole mass 0.1 kg, pole half-length 0.5 m, gravity 9.8 m/s². State: [x, ẋ, θ, θ̇]. Action: force ∈ {-10N, +10N}.

**FR-PG-CP-02:** The CartPole renderer (QPainter) shall draw: a track (horizontal line), a cart (rectangle), a pole (thick line), and a pivot point. The visual shall scale to fill the available canvas.

**FR-PG-CP-03:** A real-time state readout panel shall display x, ẋ, θ (in degrees), θ̇ with live-updating color-coded bar indicators (green within safe range, red outside).

**FR-PG-CP-04:** Episode termination conditions: |x| > 2.4 m or |θ| > 12°. On termination, the pole shall animate a "fall" (accelerate to horizontal) before reset.

**FR-PG-CP-05:** Algorithm: Q-Learning with discretized state space (binned: x=10, ẋ=10, θ=10, θ̇=10 → 10⁴ states) or REINFORCE with a 2-layer linear policy network (state_dim=4, hidden=64, action_dim=2) implemented in NumPy.

**FR-PG-CP-06:** Training controls identical to GridWorld (FR-PG-GW-05, FR-PG-GW-07, FR-PG-GW-08).

**FR-PG-CP-07:** A **pole balance duration** indicator shall display max consecutive steps held in current training run, with a trophy icon when the 200-step target is exceeded.

### 4.7 MODULE: Playground — Maze

**FR-PG-MZ-01:** The Maze environment shall be a 15×15 grid generated using Recursive Backtracking (perfect maze, guaranteed solution). Start at top-left, Goal at bottom-right.

**FR-PG-MZ-02:** Maze rendering (QPainter): walls as filled rectangles, corridors as dark paths, goal as pulsing star, agent as glowing circle.

**FR-PG-MZ-03:** Q-Learning shall train on the maze. Upon reaching goal, the optimal path shall overlay as a glowing trail (cyan line through visited cells).

**FR-PG-MZ-04:** A **path length** counter shall display current path length vs optimal (BFS shortest path), updating each episode.

**FR-PG-MZ-05:** A "Generate New Maze" button shall create a new random maze and reset training.

**FR-PG-MZ-06:** Training controls identical to GridWorld (FR-PG-GW-05, FR-PG-GW-07, FR-PG-GW-08).

### 4.8 MODULE: Comparison Arena

**FR-CA-01:** The Comparison Arena shall run Q-Learning and REINFORCE simultaneously on the same GridWorld environment (shared grid layout, independent agents).

**FR-CA-02:** The layout shall be two side-by-side GridWorld renderers (left: Q-Learning, right: REINFORCE), each with their own heatmap, arrows, and agent.

**FR-CA-03:** A shared **Learning Curve comparison** chart shall display both algorithms' episode rewards on the same axes, with distinct colors (Cyan = Q-Learning, Magenta = REINFORCE) and 50-episode moving averages.

**FR-CA-04:** A **statistics panel** shall display per-algorithm: Total Episodes, Avg Reward (last 50), Best Episode Reward, Convergence Episode (first episode exceeding 0.9 × max possible reward), Wall Time.

**FR-CA-05:** Both algorithms shall use synchronized episode counting (same episode number simultaneously) to enable fair comparison.

**FR-CA-06:** Hyperparameter panels for each algorithm shall be collapsible and independently configurable.

**FR-CA-07:** A "Start Race" button shall start both algorithms simultaneously. A "Pause/Resume" button shall pause both. A "Reset" button shall reset both.

---

## 5. Non-Functional Requirements

### 5.1 Performance

**NFR-PERF-01:** The visualization frame rate shall be ≥ 30 FPS during active training in all environments. Measured with `QElapsedTimer` over 100 consecutive frames.

**NFR-PERF-02:** UI shall remain responsive (< 16ms event loop block) during all training operations. Training shall execute in a QTimer callback, not a blocking loop.

**NFR-PERF-03:** GridWorld Q-Learning shall complete 1000 episodes in ≤ 5 seconds at max speed (6×6 grid, 100 steps/episode limit).

**NFR-PERF-04:** CartPole Q-Learning shall complete 500 episodes in ≤ 10 seconds at max speed.

**NFR-PERF-05:** Peak RAM usage shall not exceed 512 MB at any time.

### 5.2 Visual Quality

**NFR-VQ-01:** All QPainter rendering shall use `Antialiasing` and `SmoothPixmapTransform` render hints.

**NFR-VQ-02:** All animated transitions shall use easing functions (ease-in-out quadratic) — implemented as QPainter property interpolation via `QVariantAnimation` or manual lerp.

**NFR-VQ-03:** Glassmorphism panels shall achieve visual depth via: 8% opacity white fill, 1px semi-transparent border, drop shadow (offset 4px, blur 16px, alpha 40%).

**NFR-VQ-04:** Neon glow effects shall be rendered as layered radial gradients: outer glow (radius ×2.5, alpha 40%), inner glow (radius ×1.5, alpha 70%), core (radius ×1.0, alpha 100%).

### 5.3 Usability

**NFR-USE-01:** Every interactive widget shall respond to user input within 100ms.

**NFR-USE-02:** All sliders shall have visible labels with current value and units. Value changes shall immediately update the affected visualization (no "Apply" button required).

**NFR-USE-03:** All concept pages shall include a concise text label (≤ 40 chars) identifying each interactive component.

**NFR-USE-04:** Keyboard shortcuts: Space = Play/Pause training, R = Reset, 1/2/3 = switch Playground environments, C = jump to Comparison Arena.

### 5.4 Reliability

**NFR-REL-01:** The application shall not crash or raise unhandled exceptions during normal operation. All RL algorithm edge cases (division by zero in policy gradient, Q-table initialization) shall be handled.

**NFR-REL-02:** On any training error, the error state shall be displayed inline (status label) without crashing the application.

### 5.5 Maintainability

**NFR-MAINT-01:** No single source file shall exceed 600 lines. Files approaching this limit shall be split by responsibility.

**NFR-MAINT-02:** All RL algorithms shall be implemented in `core/algorithms/` with zero Qt imports. All environments shall be in `core/environments/` with zero Qt imports.

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| Agent | Decision-making entity that observes state and selects actions |
| Environment | System the agent interacts with; produces states and rewards |
| Episode | A complete sequence from initial to terminal state |
| Return Gₜ | Sum of discounted future rewards: Gₜ = Σₖ γᵏ rₜ₊ₖ₊₁ |
| Q-value | Expected return from state s taking action a: Q(s,a) = 𝔼[Gₜ \| sₜ=s, aₜ=a] |
| Policy | Mapping from states to action probabilities: π(a\|s) |
| Value function | Expected return from state s following policy π: Vπ(s) |
| Bellman equation | Recursive definition of Q-values |
| TD Error | δ = r + γ·maxQ(s',a') - Q(s,a) |
| Experience Replay | Buffer of past transitions (s,a,r,s') sampled randomly for training |
| Target Network | Slowly-updated copy of Q-network used for stable TD targets |
| REINFORCE | Monte Carlo policy gradient: update θ by ∇log π(a\|s)·Gₜ |

---

## Appendix B: Requirements Traceability Matrix

| Concept | Visualization Requirement | Algorithm Requirement | Test Coverage |
|---------|--------------------------|----------------------|---------------|
| MDP | FR-CONC-RLB-02 | core/environments/*.py | GridWorld step test |
| Discount Factor | FR-CONC-RLB-04 | γ slider → Return | Unit test: return_calc |
| Q-Table | FR-CONC-VB-01, FR-PG-GW-02 | Q-Learning update | Q-value convergence test |
| Bellman Eq. | FR-CONC-VB-02 | TD update in training | Bellman residual test |
| ε-greedy | FR-CONC-VB-03, FR-PG-GW-05 | action_select() | Exploration rate test |
| DQN Replay | FR-CONC-VB-06 | ReplayBuffer class | Buffer sample test |
| Target Net | FR-CONC-VB-07 | sync_target() | Sync interval test |
| Policy π | FR-CONC-PB-01, FR-PG-GW-09 | REINFORCE | Softmax sum = 1 test |
| REINFORCE | FR-CONC-PB-03, FR-PG-GW-09 | returns_from_rewards() | Return calculation test |
| CartPole | FR-PG-CP-01..07 | Discretized Q or linear policy | Physics step test |
| Maze | FR-PG-MZ-01..06 | Q-Learning | BFS optimality test |
| Arena | FR-CA-01..07 | Dual agent run | Parallel training test |

---

## Appendix C: Concept Coverage Map

| Lecture Concept | Module | Requirement(s) |
|-----------------|--------|----------------|
| Agent-Environment Loop | Concepts/RL Basics | FR-CONC-RLB-01 |
| MDP | Concepts/RL Basics | FR-CONC-RLB-02 |
| Markov Property | Concepts/RL Basics | FR-CONC-RLB-03 |
| Discount Factor γ | Concepts/RL Basics | FR-CONC-RLB-04, 05 |
| Return G | Concepts/RL Basics | FR-CONC-RLB-04, 05 |
| Q-Function | Concepts/Value-Based | FR-CONC-VB-01 |
| Bellman Equation | Concepts/Value-Based | FR-CONC-VB-02 |
| Q-Learning (ε-greedy) | Concepts/Value-Based | FR-CONC-VB-03 |
| Q-Learning (TD update) | Concepts/Value-Based | FR-CONC-VB-04 |
| DQN Architecture | Concepts/Value-Based | FR-CONC-VB-05 |
| Experience Replay | Concepts/Value-Based | FR-CONC-VB-06 |
| Target Network | Concepts/Value-Based | FR-CONC-VB-07 |
| Policy π(a\|s) | Concepts/Policy-Based | FR-CONC-PB-01 |
| Policy Gradient Theorem | Concepts/Policy-Based | FR-CONC-PB-02 |
| REINFORCE trajectory | Concepts/Policy-Based | FR-CONC-PB-03 |
| Policy update direction | Concepts/Policy-Based | FR-CONC-PB-04 |
| Discrete vs Continuous | Concepts/Policy-Based | FR-CONC-PB-05 |
| Atari application | Concepts/Applications | FR-CONC-APP-01 |
| AlphaGo application | Concepts/Applications | FR-CONC-APP-02 |
| Robotics application | Concepts/Applications | FR-CONC-APP-03 |
| GridWorld env | Playground | FR-PG-GW-01..10 |
| CartPole env | Playground | FR-PG-CP-01..07 |
| Maze env | Playground | FR-PG-MZ-01..06 |
| Comparison Arena | Arena | FR-CA-01..07 |
