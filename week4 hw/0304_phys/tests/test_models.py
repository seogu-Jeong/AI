import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest


def test_projectile_regression_io_shape():
    from app.ml.models import ModelFactory
    m = ModelFactory.projectile_regression()
    out = m.predict(np.zeros((5, 3)), verbose=0)
    assert out.shape == (5, 2)   # (x, y)


def test_pendulum_period_io_shape():
    from app.ml.models import ModelFactory
    m = ModelFactory.pendulum_period()
    out = m.predict(np.zeros((5, 2)), verbose=0)
    assert out.shape == (5, 1)   # T


def test_air_resistance_range_io_shape():
    from app.ml.models import ModelFactory
    m = ModelFactory.air_resistance_range()
    out = m.predict(np.zeros((5, 2)), verbose=0)
    assert out.shape == (5, 1)   # R


def test_pendulum_period_has_mape_metric():
    from app.ml.models import ModelFactory
    m = ModelFactory.pendulum_period()
    h = m.fit(np.ones((10, 2)), np.ones((10, 1)), epochs=1, verbose=0)
    assert 'mape' in h.history
