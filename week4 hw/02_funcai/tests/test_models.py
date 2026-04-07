import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pytest

ARCH_MAP = {
    'Small [32]':           [32],
    'Medium [64, 64]':      [64, 64],
    'Large [128, 128]':     [128, 128],
    'XLarge [128, 128, 64]':[128, 128, 64],
}

def test_function_approximator_output_shape():
    from app.ml.models import ModelFactory
    import numpy as np
    m = ModelFactory.function_approximator([64, 64], 'tanh', 0.01)
    out = m.predict(np.zeros((5, 1)), verbose=0)
    assert out.shape == (5, 1)

def test_function_approximator_architectures():
    from app.ml.models import ModelFactory
    for label, layers in ARCH_MAP.items():
        m = ModelFactory.function_approximator(layers, 'tanh', 0.01)
        assert m is not None

def test_param_count_small():
    from app.ml.models import ModelFactory
    n = ModelFactory.count_params([32], 'tanh')
    assert 50 < n < 500

def test_param_count_large():
    from app.ml.models import ModelFactory
    n = ModelFactory.count_params([128, 128], 'tanh')
    assert 15_000 < n < 25_000

def test_overfitting_suite_keys():
    from app.ml.models import ModelFactory
    suite = ModelFactory.overfitting_suite()
    assert set(suite.keys()) == {'underfit', 'good', 'overfit'}

def test_overfitting_suite_output_shapes():
    from app.ml.models import ModelFactory
    import numpy as np
    suite = ModelFactory.overfitting_suite()
    X = np.zeros((10, 1))
    for name, model in suite.items():
        out = model.predict(X, verbose=0)
        assert out.shape == (10, 1), f"{name} output shape wrong"
