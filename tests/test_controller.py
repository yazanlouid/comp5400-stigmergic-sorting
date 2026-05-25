"""Tests for NeuralController — forward pass, init, serialization."""

from __future__ import annotations

import numpy as np
import pytest

from src.controller import NeuralController


@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(42)


def test_forward_shape(rng):
    ctrl = NeuralController.random_init(rng)
    out = ctrl.forward(np.zeros(NeuralController.INPUTS))
    assert out.shape == (NeuralController.OUTPUTS,)


def test_softmax_sums_to_one(rng):
    ctrl = NeuralController.random_init(rng)
    out = ctrl.forward(np.zeros(NeuralController.INPUTS))
    assert abs(np.sum(out) - 1.0) < 1e-6


def test_serialize_roundtrip(rng):
    ctrl = NeuralController.random_init(rng)
    weights_list = ctrl.to_list()
    ctrl2 = NeuralController.from_list(weights_list)
    assert np.allclose(ctrl.to_list(), ctrl2.to_list())


def test_weight_count():
    assert NeuralController.TOTAL_PARAMS == 149


def test_deterministic_forward(rng):
    ctrl = NeuralController.random_init(rng)
    inp = np.array([1.0, 2.0, 0.5, 0.5, 0.0, 1.0])
    out1 = ctrl.forward(inp)
    out2 = ctrl.forward(inp)
    np.testing.assert_array_equal(out1, out2)


def test_init_pickup_bias(rng):
    ctrl = NeuralController.random_init(rng, pickup_bias=0.5)
    weights = np.array(ctrl.to_list())
    b2 = weights[144:]
    assert abs(b2[3] - 0.5) < 1e-9


def test_init_move_bias(rng):
    ctrl = NeuralController.random_init(rng, pickup_bias=0.0, move_bias=0.5)
    weights = np.array(ctrl.to_list())
    b2 = weights[144:]
    assert abs(b2[0] - 0.5) < 1e-9
    assert abs(b2[3] - 0.0) < 1e-9


def test_input_vector_from_sensor_reading(rng):
    ctrl = NeuralController.random_init(rng)
    inp = np.array([3.0, 1.0, 3.0, 1.0, 1.0, 0.0])
    out = ctrl.forward(inp)
    assert out.shape == (5,)
    assert abs(np.sum(out) - 1.0) < 1e-6
