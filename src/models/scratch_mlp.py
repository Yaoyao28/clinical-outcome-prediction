"""A minimal neural network written from scratch in NumPy — fill-in-the-blank version.

How to use this file
--------------------
Every place you need to write something is marked ``___``. Each blank has a
hint directly above it saying what shape the answer must have and which
variables are in scope. Work top to bottom; after each section, run:

    python -m pytest tests/test_scratch_mlp.py -q

and watch the pass count climb.

The whole point is the shapes. Almost every blank has exactly one arrangement
of the available arrays that produces the required shape, so if you write the
shapes down first the formula follows. Two rules cover nearly everything:

    to cancel a dimension  ->  transpose and matrix-multiply  (A.T @ B)
    to undo a broadcast    ->  sum along that axis            (X.sum(axis=0))

NumPy syntax you will need:
    A @ B            matrix multiply
    A.T              transpose
    A * B            elementwise multiply
    np.sum(A, axis=0)     sum down the rows -> shape (cols,)
    np.maximum(0, A)      elementwise max
    np.exp, np.log, np.clip, np.mean
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
class Layer(ABC):
    """One differentiable step of the network. Nothing to fill in here.

    The contract:
      forward(x)             -> output, and caches whatever backward will need
      backward(grad_output)  -> dL/d(input), and stores parameter gradients
      params() / grads()     -> parallel lists the optimizer zips together
    """

    def __init__(self) -> None:
        self.cache: dict = {}

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray: ...

    def params(self) -> list[np.ndarray]:
        return []

    def grads(self) -> list[np.ndarray]:
        return []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Linear(Layer):
    """y = x @ W + b

        x: (B, in)      B = batch size
        W: (in, out)    column j holds the weights of output neuron j
        b: (out,)       one intercept per output neuron, broadcast over the batch
        y: (B, out)

    W starts from a He-scaled normal: std = sqrt(2 / in_features). All-zeros
    would make every neuron identical forever, since identical weights get
    identical gradients.
    """

    def __init__(self, in_features: int, out_features: int, *, rng: np.random.Generator | None = None):
        super().__init__()
        rng = rng or np.random.default_rng(0)
        self.W = rng.normal(0.0, np.sqrt(2.0 / in_features), (in_features, out_features))
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Backward needs x to compute dW, and a local variable would be gone by
        # then, so stash it on the object.
        self.cache["x"] = x

        # (B, in) @ (in, out) -> (B, out), then b broadcasts across the batch.
        return ___

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """grad_output is dL/dy, shape (B, out)."""
        x = self.cache["x"]

        # dL/dW must have W's shape (in, out).
        # In scope: x is (B, in), grad_output is (B, out).
        # Only one pairing cancels B and leaves (in, out).
        self.dW = ___

        # dL/db must have b's shape (out,).
        # b was added to every one of the B rows, so every row's gradient
        # belongs to it — undo the broadcast.
        self.db = ___

        # dL/dx must have x's shape (B, in), and goes to the previous layer.
        # In scope: grad_output is (B, out), self.W is (in, out).
        return ___

    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]


class ReLU(Layer):
    """y = max(0, x), elementwise. No parameters — it only routes gradients."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache["x"] = x

        # Clamp negatives to zero, same shape in and out.
        return ___

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # The slope is 1 where the input was positive and 0 where it was not,
        # so the gradient passes through in some positions and is killed in
        # others. `self.cache["x"] > 0` is a boolean array of the right shape;
        # multiplying by it zeroes the blocked positions.
        return ___


class Sigmoid(Layer):
    """y = 1 / (1 + exp(-x)). Squashes any real number into (0, 1)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # np.exp(-x) overflows for very negative x (exp(800) is inf), so clip
        # the input first. Clipping at +-500 changes nothing: sigmoid is already
        # 0 and 1 to full float precision well before that.
        x_safe = np.clip(x, -500, 500)

        # Write the sigmoid formula using x_safe.
        y = ___

        # Backward only needs the output, not the input.
        self.cache["y"] = y
        return y

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        y = self.cache["y"]

        # dy/dx = y * (1 - y) — one of the reasons sigmoid is convenient.
        # Chain it with the incoming gradient (elementwise, not matrix, multiply).
        return ___


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class BCELoss:
    """Binary cross-entropy on probabilities.

        L = -mean( y*log(p) + (1-y)*log(1-p) )

    Reward for confidence only when it is correct: predicting 0.99 on a positive
    costs almost nothing, predicting 0.01 on one costs a lot.
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.cache: dict = {}

    def forward(self, probs: np.ndarray, targets: np.ndarray) -> float:
        # log(0) is -inf, so keep probabilities strictly inside (0, 1).
        p = np.clip(probs, self.eps, 1.0 - self.eps)
        self.cache["p"] = p
        self.cache["targets"] = targets

        # The formula above. np.mean reduces to a single number.
        return ___

    def backward(self) -> np.ndarray:
        p = self.cache["p"]
        y = self.cache["targets"]

        # Number of elements the mean divided by — the gradient carries the
        # same 1/n factor. Getting this wrong is the classic bug that only the
        # gradient check catches.
        n = p.size

        # dL/dp for the mean-reduced loss:
        #     dL/dp = (p - y) / (p * (1 - p)) / n
        # Shape matches p. Sign check: under-predicting a positive should give a
        # negative gradient, so the update pushes the probability up.
        return ___

    def __call__(self, probs: np.ndarray, targets: np.ndarray) -> float:
        return self.forward(probs, targets)


# ---------------------------------------------------------------------------
# Network and optimizer
# ---------------------------------------------------------------------------
class MLP:
    """Stack of layers built from a shape list.

        MLP([2, 16, 8, 1])  ->
            Linear(2,16) -> ReLU -> Linear(16,8) -> ReLU -> Linear(8,1) -> Sigmoid

    Every Linear gets an activation after it: ReLU for the hidden ones, Sigmoid
    for the last, so the output is a probability.
    """

    def __init__(self, shape: list[int], *, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.layers: list[Layer] = []

        for i in range(len(shape) - 1):
            self.layers.append(Linear(shape[i], shape[i + 1], rng=rng))
            is_last = i == len(shape) - 2

            # Append Sigmoid() on the last Linear, ReLU() otherwise.
            self.layers.append(___)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Feed x through the layers in order, each output becoming the next input.
        for layer in self.layers:
            x = ___
        return x

    def backward(self, grad: np.ndarray) -> None:
        # Same walk in reverse: each layer's dL/d(input) is the next one's
        # incoming gradient. `reversed(self.layers)` iterates back to front.
        for layer in reversed(self.layers):
            grad = ___

    def params(self) -> list[np.ndarray]:
        return [p for layer in self.layers for p in layer.params()]

    def grads(self) -> list[np.ndarray]:
        return [g for layer in self.layers for g in layer.grads()]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class SGD:
    """Plain gradient descent: p <- p - lr * dp"""

    def __init__(self, params: list[np.ndarray], grads: list[np.ndarray], lr: float = 0.1):
        self.params = params
        self.grads = grads
        self.lr = lr

    def step(self) -> None:
        for p, g in zip(self.params, self.grads):
            # Must modify the existing array, not rebind the name: the layer
            # holds its own reference to this array. `p -= ...` edits in place;
            # `p = p - ...` would create a new array the layer never sees.
            ___

    def zero_grad(self) -> None:
        for g in self.grads:
            g[...] = 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.1,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> list[float]:
    """Mini-batch training. Returns the mean loss for each epoch."""
    rng = rng or np.random.default_rng(0)
    loss_fn = BCELoss()
    optimizer = SGD(model.params(), model.grads(), lr=lr)

    y = y.reshape(-1, 1).astype(float)
    history: list[float] = []

    for epoch in range(epochs):
        # Reshuffle every epoch so batches differ between passes.
        order = rng.permutation(len(X))
        epoch_losses = []

        for start in range(0, len(X), batch_size):
            idx = order[start : start + batch_size]
            xb, yb = X[idx], y[idx]

            # Old gradients would otherwise accumulate into this step.
            optimizer.zero_grad()

            # 1. forward: predicted probabilities for this batch
            probs = ___

            # 2. loss: a single number measuring how wrong they are
            loss = ___

            # 3. dL/d(probs) — where backpropagation starts
            grad = ___

            # 4. push that gradient back through every layer, filling dW / db
            ___

            # 5. let the optimizer apply the update
            ___

            epoch_losses.append(loss)

        history.append(float(np.mean(epoch_losses)))
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d}  loss {history[-1]:.4f}")

    return history


# ---------------------------------------------------------------------------
# Gradient check — the part that proves the backward pass is right
# ---------------------------------------------------------------------------
def gradient_check(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    *,
    eps: float = 1e-5,
    n_samples: int = 20,
    rng: np.random.Generator | None = None,
) -> float:
    """Compare analytic gradients against finite differences.

    For one scalar parameter w, the definition of a derivative gives

        numeric = (L(w + eps) - L(w - eps)) / (2 * eps)

    which needs no calculus at all — just two forward passes. If backward() is
    right, its stored gradient matches this. Returns the largest relative error
    over ``n_samples`` randomly chosen parameters: below ~1e-6 is correct, 1e-2
    or worse means a real bug.

    This is the highest-value function in the file. Reading backprop code
    almost never reveals a missing 1/n or a flipped transpose; this does.
    """
    rng = rng or np.random.default_rng(0)
    loss_fn = BCELoss()
    y = y.reshape(-1, 1).astype(float)

    def loss_now() -> float:
        """One forward pass with the parameters as they currently stand."""
        return loss_fn(model(X), y)

    # One full pass to populate the analytic gradients.
    loss_now()
    model.backward(loss_fn.backward())

    params, grads = model.params(), model.grads()
    max_error = 0.0

    for _ in range(n_samples):
        # Pick a random scalar: first an array, then a flat index inside it.
        k = rng.integers(len(params))
        p, g = params[k], grads[k]
        flat = rng.integers(p.size)
        idx = np.unravel_index(flat, p.shape)

        original = p[idx]

        # Nudge up, measure the loss.
        p[idx] = original + eps
        loss_plus = ___

        # Nudge down, measure again.
        p[idx] = original - eps
        loss_minus = ___

        # Put the parameter back before moving on — otherwise the next sample
        # is measured against a network you have quietly modified.
        p[idx] = original

        # Central difference: the slope between the two points.
        numeric = ___

        # What backward() claimed the slope was.
        analytic = g[idx]

        # Relative error, guarded so a near-zero gradient does not divide by ~0.
        denominator = max(abs(numeric), abs(analytic), eps)
        max_error = max(max_error, abs(numeric - analytic) / denominator)

    return max_error


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def make_moons_demo() -> None:
    """Two interleaving crescents — not linearly separable.

    A network with a hidden layer should clear 95%; the same code with no
    hidden layer stalls around 85%. That gap is the argument for depth, and it
    is worth seeing on your own screen rather than taking on faith.
    """
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    model = MLP([2, 16, 8, 1], rng=np.random.default_rng(0))
    print("gradient check (max relative error):", gradient_check(model, X[:32], y[:32]))

    history = train(model, X, y, epochs=100, batch_size=32, lr=0.5, verbose=True)
    accuracy = ((model(X) > 0.5).ravel().astype(int) == y).mean()
    print(f"final loss {history[-1]:.4f}   accuracy {accuracy:.3f}")

    flat = MLP([2, 1], rng=np.random.default_rng(0))
    train(flat, X, y, epochs=100, batch_size=32, lr=0.5)
    flat_accuracy = ((flat(X) > 0.5).ravel().astype(int) == y).mean()
    print(f"no hidden layer:  accuracy {flat_accuracy:.3f}")


if __name__ == "__main__":
    make_moons_demo()
