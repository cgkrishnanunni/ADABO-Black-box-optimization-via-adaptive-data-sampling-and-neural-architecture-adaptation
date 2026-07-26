"""
Low-rank active learning (same A-opt criterion as ``active_functions``).

For scalar network output, ``I(x) = g g^T`` with ``g = ∇_θ f(x)`` (rank-1).
``F`` and ``E`` are averages of such outer products, so

    F = G_F G_F^T,   E = G_E G_E^T

with ``G_F ∈ R^{p×Nt}``, ``G_E ∈ R^{p×asymp_N}`` (``p`` = #parameters).

The A-opt score ``tr(E (F+I)^+)`` is evaluated with a small ``(Nt+1)×(Nt+1)``
pseudoinverse instead of forming or inverting any ``p×p`` matrix.

This is the acquisition used by ``PROPOSED_ALGORITHM/main.ipynb``.
"""

import torch
import numpy as np
from proposed_bbopt.util.activation_prop import activation, activation_input, activation_output


def _pack_params(weights, biases):
    """Flatten weights then biases; return vector and slice metadata for unpacking."""
    dtype = weights[0].dtype
    device = weights[0].device
    w_sizes = [int(w.numel()) for w in weights]
    b_sizes = [int(b.numel()) for b in biases]
    n_params = sum(w_sizes) + sum(b_sizes)

    theta = torch.zeros(n_params, dtype=dtype, device=device)
    w_slices = []
    k = 0
    for i, w in enumerate(weights):
        k_old = k
        k = k + w_sizes[i]
        theta[k_old:k] = w.reshape(-1)
        w_slices.append((k_old, k, w.shape))

    b_slices = []
    k_biases = k
    for i, b in enumerate(biases):
        k_old_biases = k_biases
        k_biases = k_biases + b_sizes[i]
        theta[k_old_biases:k_biases] = b.reshape(-1)
        b_slices.append((k_old_biases, k_biases))

    return theta, w_slices, b_slices


def _forward_from_theta(theta, x, hyperp, w_slices, b_slices):
    """Scalar network output given flat parameter vector (same net as I_fun)."""
    y = None
    for i in range(0, hyperp.L - 2):
        k_n_old, k_n, shape = w_slices[i]
        k_old_biases_n, k_biases_n = b_slices[i]
        W = theta[k_n_old:k_n].reshape(shape)
        bvec = theta[k_old_biases_n:k_biases_n]
        if i == 0:
            y = activation_input(torch.matmul(W, x) + bvec)
        else:
            y = y + activation(torch.matmul(W, y) + bvec)

    k_n_old, k_n, shape = w_slices[hyperp.L - 2]
    k_old_biases_n, k_biases_n = b_slices[hyperp.L - 2]
    W = theta[k_n_old:k_n].reshape(shape)
    bvec = theta[k_old_biases_n:k_biases_n]
    y = activation_output(torch.matmul(W, y) + bvec)
    return y.reshape(())


def grad_theta(x, hyperp, weights, biases, theta=None, w_slices=None, b_slices=None):
    """
    Parameter gradient g = ∇_θ f(x) for scalar network output.
    I(x) = g g^T in the dense formulation.
    """
    dtype = weights[0].dtype
    device = weights[0].device
    x = torch.as_tensor(x, dtype=dtype, device=device).reshape(-1)

    if theta is None or w_slices is None or b_slices is None:
        theta, w_slices, b_slices = _pack_params(weights, biases)

    theta_req = theta.detach().requires_grad_(True)
    y = _forward_from_theta(theta_req, x, hyperp, w_slices, b_slices)
    (g,) = torch.autograd.grad(y, theta_req)
    return g.detach()


def E_factors(hyperp, weights, biases, seed=0):
    """
    Low-rank factor G_E for E = G_E G_E^T.
    Columns are MC gradients scaled by 1/sqrt(asymp_N).
    """
    torch.manual_seed(seed)
    lo, hi = float(hyperp.normalize[0]), float(hyperp.normalize[1])
    span = hi - lo
    dtype = weights[0].dtype
    device = weights[0].device
    theta, w_slices, b_slices = _pack_params(weights, biases)

    cols = []
    scale = 1.0 / np.sqrt(hyperp.asymp_N)
    for _ in range(hyperp.asymp_N):
        x = lo + span * torch.rand(hyperp.I, dtype=dtype, device=device)
        g = grad_theta(x, hyperp, weights, biases, theta, w_slices, b_slices)
        cols.append(g * scale)
    return torch.stack(cols, dim=1)  # p × asymp_N


def F_factors(hyperp, weights, biases, Nt, X):
    """
    Low-rank factor G_F for F = G_F G_F^T.
    Columns are training-point gradients scaled by 1/sqrt(Nt).
    """
    X_cast = X.to(dtype=weights[0].dtype, device=weights[0].device)
    theta, w_slices, b_slices = _pack_params(weights, biases)

    cols = []
    scale = 1.0 / np.sqrt(Nt)
    for i in range(Nt):
        g = grad_theta(
            X_cast[:, i], hyperp, weights, biases, theta, w_slices, b_slices
        )
        cols.append(g * scale)
    return torch.stack(cols, dim=1)  # p × Nt


def a_opt_criterion(G_E, G_F, g):
    """
    tr(E (F + g g^T)^+) without forming p×p matrices.

    With B = [G_F | g], M = B B^T = F + I, and S = B^T B:
        tr(E M^+) = || S^+ (B^T G_E) ||_F^2
    when S^+ is the pseudoinverse of S.
    """
    B = torch.cat([G_F, g.reshape(-1, 1)], dim=1)
    S = B.T @ B                          # r × r, r = Nt + 1
    C = B.T @ G_E                        # r × asymp_N
    W = torch.linalg.pinv(S) @ C         # r × asymp_N
    return float(torch.sum(W * W).detach().cpu())


def E_function(hyperp, weights, biases, seed=0):
    """Return low-rank factor G_E (not the dense E matrix)."""
    return E_factors(hyperp, weights, biases, seed=seed)


def F_function(hyperp, weights, biases, Nt, X):
    """Return low-rank factor G_F (not the dense F matrix)."""
    return F_factors(hyperp, weights, biases, Nt, X)


def active_learning(hyperp, X, weights, biases, Nt, seed=0):
    """
    Propose a new normalized design point (same policy as dense active_functions):
      - low-variance coordinates: maximize min-distance to existing data
      - remaining coordinates: minimize A-opt score tr(E (F+I)^+) via low-rank algebra
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    lo, hi = float(hyperp.normalize[0]), float(hyperp.normalize[1])
    span = hi - lo

    # Number of coordinates optimized by distance (exploration subspace)
    prob = np.random.randint(low=1, high=hyperp.I)

    # Rank coordinates by |std/mean|
    X_np = X.detach().cpu().numpy()
    means = np.mean(X_np, axis=1)
    stds = np.std(X_np, axis=1)
    a = np.abs(
        np.divide(stds, means, out=np.zeros_like(stds), where=np.abs(means) > 1e-15)
    )
    ll = np.argsort(a)

    ll_dist = ll[:prob]
    ll_exploit = ll[prob:]
    X_new = X_np[ll_dist, :]

    def active_learning_optimize_first(x):
        YY = x.reshape(-1, 1) - X_new
        return -np.min(np.linalg.norm(YY, axis=0))

    # Random grid search: maximize distance to nearest neighbor in subspace
    solution = np.inf
    solution_grid = lo + span * np.random.rand(prob)
    for _ in range(hyperp.grid_opt):
        grid_search = lo + span * np.random.rand(prob)
        new_solution = active_learning_optimize_first(grid_search)
        if np.isfinite(new_solution) and new_solution < solution:
            solution = new_solution
            solution_grid = grid_search

    dtype = weights[0].dtype
    device = weights[0].device
    xd = torch.tensor(solution_grid, dtype=dtype, device=device)

    # Low-rank factors (independent of the exploitation grid point)
    G_F = F_factors(hyperp, weights, biases, Nt, X)
    G_E = E_factors(hyperp, weights, biases, seed=seed)
    theta, w_slices, b_slices = _pack_params(weights, biases)

    def active_learning_optimize(x):
        xx = torch.zeros(hyperp.I, dtype=dtype, device=device)
        xx[ll_dist] = xd
        if x.numel() > 0:
            xx[ll_exploit] = x
        g = grad_theta(xx, hyperp, weights, biases, theta, w_slices, b_slices)
        return a_opt_criterion(G_E, G_F, g)

    # Random grid search over the complementary coordinates
    n_exploit = int(hyperp.I - prob)
    solution = np.inf
    if n_exploit > 0:
        solution_grid = torch.tensor(
            lo + span * np.random.rand(n_exploit), dtype=dtype, device=device
        )
        for _ in range(hyperp.grid_opt):
            grid_search = torch.tensor(
                lo + span * np.random.rand(n_exploit), dtype=dtype, device=device
            )
            new_solution = active_learning_optimize(grid_search)
            if np.isfinite(new_solution) and new_solution < solution:
                solution = new_solution
                solution_grid = grid_search
    else:
        solution_grid = torch.zeros(0, dtype=dtype, device=device)

    out = torch.zeros(hyperp.I, dtype=dtype, device=device)
    out[ll_dist] = xd
    if n_exploit > 0:
        out[ll_exploit] = solution_grid
    return out
