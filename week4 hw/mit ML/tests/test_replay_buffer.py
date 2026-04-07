import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from core.algorithms.replay_buffer import ReplayBuffer


def test_replay_buffer_push_and_len():
    rb = ReplayBuffer(capacity=10)
    rb.push(0, 1, 0.5, 1, False)
    assert len(rb) == 1


def test_replay_buffer_sample_correct_shape():
    rb = ReplayBuffer(capacity=20)
    for i in range(15):
        rb.push(i, 0, -0.01, i + 1, False)
    states, actions, rewards, next_states, dones = rb.sample(5)
    assert states.shape == (5,)
    assert rewards.shape == (5,)


def test_replay_buffer_raises_on_insufficient():
    rb = ReplayBuffer(capacity=10)
    rb.push(0, 0, 0.0, 1, False)
    with pytest.raises(ValueError):
        rb.sample(5)


def test_replay_buffer_circular_overwrite():
    rb = ReplayBuffer(capacity=5)
    for i in range(10):
        rb.push(i, 0, 0.0, i + 1, False)
    assert len(rb) == 5


def test_replay_buffer_fill_ratio():
    rb = ReplayBuffer(capacity=10)
    for i in range(5):
        rb.push(i, 0, 0.0, i + 1, False)
    assert rb.fill_ratio == pytest.approx(0.5)
