# TRD: Technical Architecture for MLP XOR Solver Web App

## 1. System Overview
The MLP XOR Solver Web App is a client-side heavy single-page application (SPA). It implements the Multi-Layer Perceptron logic from `04_mlp_numpy.py` using JavaScript/TypeScript to provide real-time training and visualization without requiring a server-side runtime.

## 2. Tech Stack
- **Frontend Framework:** React 18+ (for state management and UI components).
- **Styling:** Tailwind CSS (for rapid UI development and responsiveness).
- **Math Engine:** 
    - **Option A:** Pure JavaScript with `mathjs` for matrix operations.
    - **Option B:** `TensorFlow.js` for GPU-accelerated training.
- **Visualization:**
    - **Chart.js:** For the loss curve.
    - **D3.js or Plotly.js:** For the 2D contour plot (decision boundary).
    - **Canvas API:** For the network topology and weight visualization.

## 3. Mathematical Engine Implementation
The core logic will be ported from the Python class `MLP` in `04_mlp_numpy.py`.

### 3.1 State Management
The `MLP` state will be stored in a React Context or a specialized Store (e.g., Zustand).
```typescript
interface MLPState {
  weights: [number[][], number[][]]; // W1, W2
  biases: [number[][], number[][]];  // b1, b2
  learningRate: number;
  activation: 'sigmoid' | 'relu' | 'tanh';
  lossHistory: number[];
}
```

### 3.2 Matrix Operations
We will implement a `Matrix` utility class to mimic NumPy's `@` (dot product) and element-wise operations.
- `dot(A, B)`: Standard matrix multiplication.
- `add(A, B)`: Element-wise addition.
- `applyFunc(A, f)`: Apply activation function $f$ to each element.

## 4. Training Loop Optimization
To prevent the UI thread from freezing during training, the training loop will run in a **Web Worker**.

### 4.1 Web Worker Communication
- **Main Thread -> Worker:** `start_training(params)`, `update_params(lr)`, `stop_training()`.
- **Worker -> Main Thread:** `epoch_completed(current_weights, current_loss, current_activations)`.

### 4.2 Frame Rate Control
The Worker will send updates every $N$ epochs (where $N$ is configurable) to maintain 60 FPS in the UI.

## 5. Visualization Logic

### 5.1 Decision Boundary (Contour Plot)
- **Grid Generation:** Generate a $100 \times 100$ grid of coordinates between $[-0.5, 1.5]$.
- **Inference:** Run the current weights through the forward pass for all 10,000 grid points.
- **Rendering:** Use `Plotly.js`'s contour plot type or a custom WebGL shader for maximum performance.

### 5.2 Hidden Layer Activation Heatmap
- Mirror the logic from `04_mlp_numpy.py`:
- `activations = forward_hidden(X_xor)`
- Render a $4 \times N_{hidden}$ grid using SVG or Canvas, where cell color is mapped to activation value $[0, 1]$.

## 6. Component Architecture
```text
App
├── Header
├── Sidebar (Controls)
│   ├── LearningRateSlider
│   ├── HiddenNeuronInput
│   ├── ActivationSelector
│   └── ActionButtons (Play/Pause/Reset)
├── MainContent
│   ├── DecisionBoundaryPlot (Plotly)
│   └── StatsDisplay
└── Footer
    ├── LossChart (Chart.js)
    └── ActivationHeatmap
```

## 7. Mathematical Formulas (Implementation Reference)
The JavaScript implementation must strictly follow these formulas from the reference script:
1. **Sigmoid:** $1 / (1 + \exp(-x))$ with clipping $[-500, 500]$.
2. **Forward Layer 1:** $z_1 = XW_1 + b_1, a_1 = \sigma(z_1)$.
3. **Forward Layer 2:** $z_2 = a_1W_2 + b_2, a_2 = \sigma(z_2)$.
4. **Backward (Gradients):**
    - $dz_2 = a_2 - y$
    - $dW_2 = a_1^T \cdot dz_2$
    - $dz_1 = (dz_2 \cdot W_2^T) \odot \sigma'(z_1)$
    - $dW_1 = X^T \cdot dz_1$

## 8. Data Persistence
- **LocalStorage:** Save the last used hyperparameters and random seed so the user returns to the same state after a refresh.
- **Export/Import:** Allow users to download weights as a JSON file and re-upload them to resume training.

## 9. Testing Strategy
- **Unit Tests:** Verify matrix operations (dot product, transpose) against known values from NumPy.
- **Integration Tests:** Ensure the Web Worker correctly processes 10,000 epochs and returns consistent results.
- **Performance Profiling:** Monitor memory leaks in the Web Worker during long-running training sessions.

## 10. Security & Deployment
- **Deployment:** Vercel or GitHub Pages (Static Hosting).
- **Security:** No backend means no API security risks. Ensure `eval()` is not used for custom activation functions to prevent XSS.

---
*End of TRD*
