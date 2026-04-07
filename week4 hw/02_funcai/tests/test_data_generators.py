import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from app.ml.data_generators import DataGenerators

# ── MOD-01 ────────────────────────────────────────────────────────────────────

def test_function_approx_sin_shapes():
    X_tr, y_tr, X_te, y_te = DataGenerators.function_approximation('sin(x)')
    assert X_tr.shape == (200, 1)  # FR-MOD01-08: 200 train
    assert y_tr.shape == (200, 1)
    assert X_te.shape == (400, 1)  # FR-MOD01-08: 400 test
    assert y_te.shape == (400, 1)

def test_function_approx_domain_f01():
    X_tr, _, _, _ = DataGenerators.function_approximation('sin(x)')
    assert np.isclose(X_tr.min(), -2 * np.pi, atol=0.01)
    assert np.isclose(X_tr.max(),  2 * np.pi, atol=0.01)

def test_function_approx_f04_domain_and_size():
    # FR-MOD01-13: F-04 uses 500 train points over [-3π, 3π]
    X_tr, y_tr, X_te, y_te = DataGenerators.function_approximation('extreme')
    assert X_tr.shape == (500, 1)
    assert np.isclose(X_tr.min(), -3 * np.pi, atol=0.01)
    assert np.isclose(X_tr.max(),  3 * np.pi, atol=0.01)

def test_function_approx_sin_values():
    X_tr, y_tr, _, _ = DataGenerators.function_approximation('sin(x)')
    np.testing.assert_allclose(y_tr, np.sin(X_tr), atol=1e-10)

# ── MOD-03 ────────────────────────────────────────────────────────────────────

def test_overfitting_shapes():
    # FR-MOD03-01: 100 train, 50 val, 200 test (noise-free)
    X_tr, y_tr, X_val, y_val, X_te, y_te = DataGenerators.overfitting(
        n_train=100, n_val=50, noise=0.3)
    assert X_tr.shape  == (100, 1)
    assert X_val.shape == (50,  1)
    assert X_te.shape  == (200, 1)

def test_overfitting_test_is_noisefree():
    # FR-MOD03-01: test data is noise-free
    X_te = np.linspace(-2, 2, 200).reshape(-1, 1)
    f = lambda x: np.sin(2 * x) + 0.5 * x
    _, _, _, _, X_te_gen, y_te_gen = DataGenerators.overfitting(n_train=100, n_val=50, noise=0.3)
    np.testing.assert_allclose(y_te_gen, f(X_te_gen), atol=1e-10)
