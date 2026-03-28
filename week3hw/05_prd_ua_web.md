# PRD: Universal Approximation Theorem Interactive Lab

## 1. Project Overview
This project is a web-based interactive laboratory that visualizes the **Universal Approximation Theorem (Cybenko, 1989)**. Based on the `05_universal_approximation.py` script, it allows users to witness how a single hidden layer neural network can approximate any continuous function given enough neurons. The web app will emphasize the relationship between "network width" (number of neurons) and "approximation accuracy".

## 2. Problem Statement
The Universal Approximation Theorem is often taught as a dry mathematical proof. Students struggle to visualize how a sum of sigmoid or tanh "bumps" can create a complex sine wave or a step function. Static scripts only show pre-defined results; an interactive tool is needed to allow experimentation.

## 3. Product Goals
- Provide a "Sandbox" where users can select or draw target functions.
- Demonstrate the impact of neuron count ($N$) on the Mean Squared Error (MSE).
- Visualize the contribution of individual hidden neurons to the final output.

## 4. User Personas
### Persona: Researcher Rick
- **Background:** Graduate student in Mathematics.
- **Goal:** Verify the limits of the theorem with highly oscillatory functions.
- **Requirement:** High precision plots and the ability to add up to 500 neurons.

### Persona: Beginner Beth
- **Background:** Data Science bootcamp student.
- **Goal:** Understand why deep learning models need so many parameters.
- **Requirement:** Easy-to-understand visual metaphors and guided presets.

## 5. Functional Requirements

### 5.1 Function Selection & Generation
- **Preset Functions:** Sine, Step, Sawtooth, Square, and Complex (Sum of Sines).
- **Custom Function Builder:** A "Draw" mode where users can use their mouse to draw a custom 1D continuous function on a coordinate plane.
- **Input Range:** $x \in [0, 1]$.

### 5.2 Real-time Approximation Engine
- **Neuron Slider:** Adjust the number of hidden neurons from 1 to 100 in real-time.
- **Activation Selector:** Tanh, Sigmoid, ReLU.
- **Instant Training:** As the slider moves, the model should ideally update its approximation (using pre-trained weights or very fast local training).

### 5.3 Comparative Visualization
- **Multi-Plot View:** Compare the "True Function" vs. the "NN Approximation" (as seen in `05_universal_approximation.py`).
- **Residual Plot:** A plot showing the error $|f(x) - \hat{f}(x)|$ across the domain.
- **Individual Neuron View:** Toggle a view that shows the output of *each* hidden neuron before they are summed.

### 5.4 Educational Tooltips
- Display the theorem's formula: $|f(x) - \sum w_i \sigma(v_i x + b_i)| < \epsilon$.
- Dynamic explanation: "With 3 neurons, the model only captures the basic trend. With 50, it matches the frequency."

## 6. UI/UX Design Requirements
- **Interactive Graph:** Main area features a large interactive SVG/Canvas graph.
- **Control Side-Panel:** Fixed panel for function selection and sliders.
- **Mathematical Aesthetic:** Use LaTeX rendering (KaTeX) for all formulas.
- **Clean Layout:** Minimalist design to focus the user's attention on the curve fitting.

## 7. Interaction Flow
1. User selects "Sine Wave" from the menu.
2. The blue "True Function" appears on the graph.
3. User drags the "Neuron Count" slider to **3**.
4. The red "NN Approximation" line appears, showing a very poor fit.
5. User drags the slider to **50**.
6. The red line snaps to the blue line, and a "Success!" badge appears with the MSE value.

## 8. Technical Constraints
- **Responsiveness:** Updates must happen in < 50ms for the "Width" slider to feel "alive".
- **Mobile Support:** Touch support for drawing custom functions.
- **Browser Support:** Modern browsers (ES6+).

## 9. Edge Cases
- **Discontinuous Functions:** The Step function is a challenge for continuous activation functions like Tanh. Highlight this limitation.
- **Extrapolation:** Show what happens when $x$ goes outside $[0, 1]$. (The model will likely fail).
- **Overfitting:** Allow users to add noise to the target function to see how a high neuron count might "fit the noise".

## 10. Success Criteria
- Users can successfully approximate a "Hand-drawn" function with < 0.05 MSE using 50+ neurons.
- The app is adopted by at least one university AI course as a demo.

## 11. Appendix: Mathematical Context
The app will reference Cybenko's paper (1989) and Hornik's work (1991) to provide academic depth for advanced users.

---
*End of PRD*
