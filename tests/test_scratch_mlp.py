"""Tests for the from-scratch NumPy MLP.

These are written before the implementation on purpose: they define what
"correct" means, so you can run them while filling in the TODOs and watch them
go green one at a time. Run only this file while working:

    python -m pytest tests/test_scratch_mlp.py -x -q

``-x`` stops at the first failure, which is what you want here.

Suggested order: Linear shapes → ReLU → Sigmoid → BCELoss → MLP wiring →
SGD → gradient check → train → the moons test last.
"""

import numpy as np
import pytest

from src.models.scratch_mlp import (
    MLP,
    SGD,
    BCELoss,
    Linear,
    ReLU,
    Sigmoid,
    gradient_check,
    train,
)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
def test_linear_forward_shape_and_value():
    layer = Linear(3, 2)
    layer.W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    layer.b = np.array([0.5, -0.5])

    x = np.array([[1.0, 2.0, 3.0]])
    out = layer.forward(x)

    assert out.shape == (1, 2)
    np.testing.assert_allclose(out, [[1 + 3 + 0.5, 2 + 3 - 0.5]])


def test_linear_backward_shapes():
    layer = Linear(4, 3)
    x = np.random.default_rng(0).normal(size=(8, 4))
    layer.forward(x)

    grad_in = layer.backward(np.ones((8, 3)))

    assert grad_in.shape == x.shape
    assert layer.dW.shape == layer.W.shape
    assert layer.db.shape == layer.b.shape


def test_linear_bias_gradient_sums_over_batch():
    """b is broadcast across the batch, so its gradient sums over it."""
    layer = Linear(2, 3)
    layer.forward(np.zeros((5, 2)))
    layer.backward(np.ones((5, 3)))
    np.testing.assert_allclose(layer.db, np.full(3, 5.0))


def test_relu_zeroes_negatives_both_ways():
    relu = ReLU()
    x = np.array([[-2.0, -0.5, 0.0, 1.5]])

    out = relu.forward(x)
    np.testing.assert_allclose(out, [[0.0, 0.0, 0.0, 1.5]])

    grad_in = relu.backward(np.ones_like(x))
    np.testing.assert_allclose(grad_in, [[0.0, 0.0, 0.0, 1.0]])


def test_sigmoid_values_and_derivative():
    sigmoid = Sigmoid()
    out = sigmoid.forward(np.array([[0.0]]))
    np.testing.assert_allclose(out, [[0.5]])

    grad_in = sigmoid.backward(np.array([[1.0]]))
    np.testing.assert_allclose(grad_in, [[0.25]])  # 0.5 * (1 - 0.5)


def test_sigmoid_does_not_overflow():
    out = Sigmoid().forward(np.array([[-800.0, 800.0]]))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, [[0.0, 1.0]], atol=1e-10)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def test_bce_loss_value():
    loss_fn = BCELoss()
    probs = np.array([[0.9], [0.1]])
    targets = np.array([[1.0], [0.0]])

    expected = -np.mean([np.log(0.9), np.log(0.9)])
    assert loss_fn(probs, targets) == pytest.approx(expected)


def test_bce_loss_handles_certainty_without_nan():
    loss_fn = BCELoss()
    value = loss_fn(np.array([[0.0], [1.0]]), np.array([[1.0], [0.0]]))
    assert np.isfinite(value)


def test_bce_backward_shape_and_sign():
    loss_fn = BCELoss()
    probs = np.array([[0.8], [0.3]])
    targets = np.array([[1.0], [0.0]])
    loss_fn(probs, targets)

    grad = loss_fn.backward()

    assert grad.shape == probs.shape
    assert grad[0, 0] < 0  # under-predicting a positive: push the probability up
    assert grad[1, 0] > 0  # over-predicting a negative: push it down


# ---------------------------------------------------------------------------
# Network wiring
# ---------------------------------------------------------------------------
def test_mlp_layer_structure():
    model = MLP([4, 8, 1])
    kinds = [type(layer).__name__ for layer in model.layers]
    assert kinds == ["Linear", "ReLU", "Linear", "Sigmoid"]


def test_mlp_forward_returns_probabilities():
    model = MLP([3, 6, 1])
    out = model(np.random.default_rng(0).normal(size=(10, 3)))

    assert out.shape == (10, 1)
    assert ((out >= 0) & (out <= 1)).all()


def test_mlp_params_and_grads_align():
    model = MLP([3, 5, 2, 1])
    params, grads = model.params(), model.grads()

    assert len(params) == len(grads) == 6  # W and b for each of 3 Linear layers
    assert all(p.shape == g.shape for p, g in zip(params, grads))


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
def test_sgd_updates_in_place():
    w = np.array([[1.0, 2.0]])
    g = np.array([[0.5, -0.5]])
    original = w  # same object

    SGD([w], [g], lr=0.1).step()

    np.testing.assert_allclose(w, [[0.95, 2.05]])
    assert w is original, "update must be in place or layers keep the old array"


def test_sgd_zero_grad():
    g = np.ones((2, 2))
    SGD([np.zeros((2, 2))], [g], lr=0.1).zero_grad()
    np.testing.assert_allclose(g, np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# The one that matters
# ---------------------------------------------------------------------------
def test_gradient_check_passes():
    """Analytic gradients must match finite differences."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 4))
    y = rng.integers(0, 2, 16)

    model = MLP([4, 8, 4, 1], rng=rng)
    max_error = gradient_check(model, X, y, n_samples=30, rng=rng)

    assert max_error < 1e-6, f"backward pass is wrong: max relative error {max_error:.2e}"


def test_gradient_check_catches_a_broken_backward(monkeypatch):
    """Sanity check on the check itself: break backward, the error must explode."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, 3))
    y = rng.integers(0, 2, 16)
    model = MLP([3, 6, 1], rng=rng)

    original = Linear.backward

    def broken(self, grad_output):
        grad_in = original(self, grad_output)
        self.dW *= 0.5  # plausible-looking bug: a stray factor
        return grad_in

    monkeypatch.setattr(Linear, "backward", broken)
    assert gradient_check(model, X, y, n_samples=30, rng=rng) > 1e-3


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------
def test_training_reduces_loss_on_a_separable_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    model = MLP([3, 8, 1], rng=rng)
    history = train(model, X, y, epochs=40, batch_size=32, lr=0.5, rng=rng)

    assert len(history) == 40
    assert history[-1] < history[0] * 0.5


def test_hidden_layer_beats_no_hidden_layer_on_moons():
    """The argument for hidden layers, as an assertion.

    Two crescents are not linearly separable, so a network with no hidden layer
    is stuck near 85% while one hidden layer clears 90%.
    """
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=800, noise=0.2, random_state=42)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    rng = np.random.default_rng(0)

    deep = MLP([2, 16, 8, 1], rng=np.random.default_rng(0))
    train(deep, X, y, epochs=150, batch_size=32, lr=0.5, rng=rng)
    deep_accuracy = ((deep(X) > 0.5).ravel().astype(int) == y).mean()

    flat = MLP([2, 1], rng=np.random.default_rng(0))
    train(flat, X, y, epochs=150, batch_size=32, lr=0.5, rng=rng)
    flat_accuracy = ((flat(X) > 0.5).ravel().astype(int) == y).mean()

    assert deep_accuracy > 0.90
    assert deep_accuracy > flat_accuracy + 0.03
