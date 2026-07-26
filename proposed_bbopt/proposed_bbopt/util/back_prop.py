"""
Forward and adjoint (backpropagation) passes through the residual surrogate.

Used for:
- evaluating / training metrics (``forward_prop``)
- spectral layer-insertion (``adjoint_prop`` + ``spectral_decomp``)
"""

import torch
import numpy as np


def _t(x, dtype=None):
    """Ensure torch tensor; optionally cast to dtype (avoids float/double matmul errors)."""
    t = x if torch.is_tensor(x) else torch.as_tensor(x)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype=dtype)
    return t


def forward_prop(hyperp, Nt, O, X, activation_input, activation, activation_output, weights, biases):
    """
    Forward residual network over all Nt samples.

    X_state[i, s, :] = hidden state of residual block i for sample s
    output[:, s]     = network output for sample s
    """
    # Use weight dtype as reference (SGD returns float64; new layers may be float32)
    dtype = _t(weights[0]).dtype
    X = _t(X, dtype)
    X_state = torch.zeros((hyperp.L - 2, Nt, hyperp.Nh), dtype=dtype)
    output = torch.zeros((O, Nt), dtype=dtype)

    # Layer 0: activation_input(W0 x + b0) for all samples (cols of X)
    W0 = _t(weights[0], dtype)
    b0 = _t(biases[0], dtype).reshape(-1, 1)
    X_state[0] = activation_input(torch.matmul(W0, X) + b0).T

    # Residual blocks i = 1 .. L-3
    for i in range(1, hyperp.L - 2):
        Wi = _t(weights[i], dtype)
        bi = _t(biases[i], dtype).reshape(-1, 1)
        prev = X_state[i - 1].T  # (Nh, Nt)
        X_state[i] = X_state[i - 1] + activation(torch.matmul(Wi, prev) + bi).T

    # Output layer
    Wout = _t(weights[hyperp.L - 2], dtype)
    bout = _t(biases[hyperp.L - 2], dtype).reshape(-1, 1)
    output = activation_output(
        torch.matmul(Wout, X_state[hyperp.L - 3].T) + bout
    )

    return X_state, output


def adjoint_prop(
    hyperp, Nt, O, Y,
    activation_input_derivative, activation_derivative, activation_output_derivative,
    output, X_state, weights, biases,
):
    """
    Adjoint (backward) states for all samples.

    Same recurrence as the per-sample loop, evaluated in batch over Nt.
    """
    dtype = _t(weights[0]).dtype
    Y = _t(Y, dtype)
    output = _t(output, dtype)
    X_state = _t(X_state, dtype)

    lambda_state = torch.zeros((hyperp.L - 2, Nt, hyperp.Nh), dtype=dtype)
    lambda_output = torch.zeros((O, Nt), dtype=dtype)

    # λ_out = 2 (y - f) / (Nt O)   (Y is length-Nt when O=1)
    if Y.ndim == 1:
        Y_mat = Y.reshape(O, Nt)
    else:
        Y_mat = Y
    lambda_output = (2 * (Y_mat - output)) / (Nt * O)

    # Last residual block (i = L-3): uses output-layer derivative
    i = hyperp.L - 3
    W = _t(weights[i + 1], dtype)
    b = _t(biases[i + 1], dtype).reshape(-1, 1)
    pre = torch.matmul(W, X_state[i].T) + b
    second = activation_output_derivative(pre)
    net = second * lambda_output
    lambda_state[i] = torch.matmul(W.T, net).T

    # Earlier residual blocks i < L-3 (descending: L-4, ..., 0)
    for i in reversed(range(0, hyperp.L - 3)):
        W = _t(weights[i + 1], dtype)
        b = _t(biases[i + 1], dtype).reshape(-1, 1)
        pre = torch.matmul(W, X_state[i].T) + b
        second = activation_derivative(pre)
        if not torch.is_tensor(second):
            second = torch.as_tensor(second, dtype=dtype, device=lambda_state.device)
        else:
            second = second.to(dtype=dtype)
        net = second * lambda_state[i + 1].T
        lambda_state[i] = lambda_state[i + 1] + torch.matmul(W.T, net).T

    return lambda_state, lambda_output


