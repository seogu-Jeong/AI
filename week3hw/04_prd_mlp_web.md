# PRD: Interactive MLP XOR Visualization Web Platform

## 1. Executive Summary
The goal of this project is to transform the `04_mlp_numpy.py` script into a high-performance, interactive web-based educational tool. This platform will allow users to visualize the training process of a Multi-Layer Perceptron (MLP) solving the non-linear XOR problem in real-time. Unlike a static script, the web version will focus on interactivity, allowing users to "feel" the impact of hyperparameter changes on the decision boundary and hidden layer activations.

## 2. Product Vision & Core Value
- **Vision:** Demystify the "black box" of neural networks through real-time visual feedback.
- **Core Value:** Providing a tactile learning experience where abstract concepts like backpropagation, weight updates, and non-linear separation become visible and intuitive.

## 3. Target Audience
- **Primary:** Computer Science and AI undergraduate students.
- **Secondary:** Educators looking for visual aids for deep learning lectures.
- **Tertiary:** Self-taught developers transitioning into Data Science.

## 4. User Personas
### Persona A: Student Sam
- **Background:** Just learned about the XOR problem's historical significance.
- **Needs:** Wants to see *how* the decision boundary curves over time.
- **Pain Point:** Static plots in textbooks don't show the dynamics of learning.

### Persona B: Instructor Iris
- **Background:** Teaches Introduction to Machine Learning.
- **Needs:** Needs a tool that can demonstrate the effect of different learning rates quickly without restarting a script.
- **Pain Point:** Setting up environments for students is time-consuming.

## 5. Functional Requirements

### 5.1 Real-time Training Dashboard
- **Live Loss Curve:** A chart that updates every 10-100 epochs showing the MSE (Mean Squared Error).
- **Dynamic Decision Boundary:** A contour plot that shifts and warps as the network trains.
- **Epoch Control:** Buttons to Start, Pause, Reset, and Step-through (single epoch) training.

### 5.2 Interactive Hyperparameter Tuning
- **Learning Rate ($\eta$) Slider:** Range [0.001, 1.0].
- **Hidden Layer Size Selector:** Options for 2, 4, 8, or 16 neurons.
- **Activation Function Toggle:** Sigmoid, Tanh, ReLU.
- **Initialization Seed:** Ability to change the random seed to see how different initial weights affect convergence.

### 5.3 Advanced Visualization Features
- **Hidden Layer Heatmap:** Visualizing the activation levels of hidden neurons for all 4 XOR inputs simultaneously (Mirroring the `04_mlp_numpy.py` subplot).
- **Weight Magnitude Links:** A diagram of the network where edge thickness represents the absolute value of the weights.
- **Gradient Flow:** A visual indicator of how large the gradients are during backprop.

## 6. UI/UX Requirements
- **Layout:** Three-column layout.
    - **Left:** Controls & Configuration.
    - **Center:** The Decision Boundary Plot (Primary focus).
    - **Right:** Loss Curve and Hidden Activation Heatmap.
- **Responsiveness:** Must work on Desktop (Chrome/Safari) and Tablet (iPad).
- **Aesthetics:** "Dark Mode" by default to match modern developer tool aesthetics.

## 7. User Interaction Scenarios
1. **Scenario 1: Observing Convergence**
    - User clicks "Train".
    - The decision boundary starts as a flat plane.
    - As epochs pass, the loss drops, and the boundary warps into two distinct regions.
    - User pauses when the loss hits 0.01.

2. **Scenario 2: Breaking the Model**
    - User sets learning rate to 5.0 (Extreme).
    - Training starts.
    - The loss curve fluctuates wildly (instability).
    - The decision boundary flickers erratically.
    - User learns the importance of a stable learning rate.

## 8. Non-Functional Requirements
- **Performance:** Training should run at 60fps for the first 1000 epochs to ensure smooth animation.
- **Accessibility:** Color-blind friendly palettes for the contour plots (e.g., Viridis or Magma).
- **No-Install:** Must run entirely in the browser without a backend server requirement (using client-side JS/Wasm).

## 9. Edge Cases & Error Handling
- **Vanishing Gradient:** If Sigmoid is used with huge initial weights, show a "Vanishing Gradient" warning.
- **Divergence:** If the loss becomes `NaN` due to a high learning rate, automatically stop training and suggest a lower LR.
- **Numerical Stability:** Handle potential overflows in the exponent of the Sigmoid function (as seen in the Python script's `np.clip`).

## 10. Success Metrics
- **Engagement:** Time spent per session > 5 minutes.
- **Education:** 90% of surveyed users report a better understanding of the XOR problem after use.
- **Load Time:** App should be interactive within 2 seconds.

## 11. Future Roadmap
- Support for custom datasets (User can click to add points).
- Export training animation as a GIF.
- Multi-layer support (Deep MLP).

---
*End of PRD*
