from __future__ import annotations
import numpy as np


class LinearPolicy:
    """Linear softmax policy: π(a|s) = softmax(W[s] + b)."""

    def __init__(self, n_states: int, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions
        self.W = np.zeros((n_states, n_actions))
        self.b = np.zeros(n_actions)

    def forward(self, state: int) -> np.ndarray:
        """Return softmax probabilities for state."""
        logits = self.W[state] + self.b
        logits -= np.max(logits)  # numerical stability
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def gradient(self, state: int, action: int) -> tuple[np.ndarray, np.ndarray]:
        """∇ log π(a|s) = one_hot(a) - π(·|s) for linear softmax."""
        pi = self.forward(state)
        one_hot = np.zeros(self.n_actions)
        one_hot[action] = 1.0
        grad = one_hot - pi
        dW = np.zeros_like(self.W)
        dW[state] = grad
        db = grad
        return dW, db


class REINFORCE:
    """Monte-Carlo policy gradient (REINFORCE algorithm)."""

    def __init__(self, policy: LinearPolicy, gamma: float = 0.99, lr: float = 0.01):
        self.policy = policy
        self.gamma = gamma
        self.lr = lr

    def select_action(self, state: int) -> int:
        """Sample action from policy."""
        probs = self.policy.forward(state)
        return int(np.random.choice(self.policy.n_actions, p=probs))

    def compute_returns(self, rewards: list[float]) -> np.ndarray:
        """Compute discounted returns Gₜ = Σ γᵏ rₜ₊ₖ (backward pass)."""
        G = np.zeros(len(rewards))
        cumulative = 0.0
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + self.gamma * cumulative
            G[t] = cumulative
        return G

    def finish_episode(
        self,
        states: list[int],
        actions: list[int],
        rewards: list[float],
    ) -> None:
        """Compute returns, normalize, then do gradient ascent."""
        G = self.compute_returns(rewards)
        # Normalize
        if len(G) > 1:
            std = np.std(G) + 1e-8
            G = (G - np.mean(G)) / std

        # Accumulate gradients
        dW = np.zeros_like(self.policy.W)
        db = np.zeros_like(self.policy.b)
        for t, (s, a, g) in enumerate(zip(states, actions, G)):
            gW, gb = self.policy.gradient(s, a)
            dW += g * gW
            db += g * gb

        # Gradient ascent
        self.policy.W += self.lr * dW
        self.policy.b += self.lr * db
