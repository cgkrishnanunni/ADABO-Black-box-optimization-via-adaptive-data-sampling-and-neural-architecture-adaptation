"""
Active-learning acquisition (alternative to Cuckoo Search on the surrogate).

Dense Gram-matrix version of active learning. The optimizer uses the
low-rank implementation in ``active_functions_lowrank`` by default.
"""

import torch
import numpy as np
from proposed_bbopt.util.I_fun import I_fun


def E_function(hyperp, weights, biases, seed=0):
    """Monte Carlo average of I(x) over asymp_N uniform samples in [normalize]."""
    torch.manual_seed(seed)
    lo, hi = float(hyperp.normalize[0]), float(hyperp.normalize[1])
    span = hi - lo
    dtype = weights[0].dtype
    device = weights[0].device
    E = 0
    for _ in range(hyperp.asymp_N):
        x = lo + span * torch.rand(hyperp.I, dtype=dtype, device=device)
        E = E + I_fun(x, hyperp, weights, biases)
    return E / hyperp.asymp_N


def F_function(hyperp, weights, biases, Nt, X):
    """Average of I(x) over the current training inputs X."""
    X_cast = X.to(dtype=weights[0].dtype, device=weights[0].device)
    F = 0
    for i in range(Nt):
        F = F + I_fun(X_cast[:, i], hyperp, weights, biases)
    return F / Nt


def active_learning(hyperp, X, weights, biases, Nt, seed=0):
    """
    Propose a new normalized design point:
      - low-variance coordinates: maximize min-distance to existing data
      - remaining coordinates: minimize A-opt style criterion tr(E (F+I)^+)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    lo, hi = float(hyperp.normalize[0]), float(hyperp.normalize[1])
    span = hi - lo

    # Number of coordinates optimized by distance (exploration subspace)
    prob = np.random.randint(low=1, high=hyperp.I)

    # Rank coordinates by |std/mean| (same formula, vectorized over features)
    X_np = X.detach().cpu().numpy()
    means = np.mean(X_np, axis=1)
    stds = np.std(X_np, axis=1)
    # Avoid 0/0 when a feature is constant (e.g. all zeros)
    a = np.abs(np.divide(stds, means, out=np.zeros_like(stds), where=np.abs(means) > 1e-15))
    ll = np.argsort(a)

    ll_dist = ll[:prob]       # coordinates filled by distance search
    ll_exploit = ll[prob:]    # coordinates filled by A-opt search
    X_new = X_np[ll_dist, :]  # shape (prob, Nt)

    def active_learning_optimize_first(x):
        # Same as column-wise (x - X_new[:, i]) then -min ||.||_2
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

    # Precompute E, F once (independent of the exploitation grid point)
    F = F_function(hyperp, weights, biases, Nt, X)
    E = E_function(hyperp, weights, biases, seed=seed)

    def active_learning_optimize(x):
        xx = torch.zeros(hyperp.I, dtype=dtype, device=device)
        xx[ll_dist] = xd
        if x.numel() > 0:
            xx[ll_exploit] = x
        val = torch.trace(
            torch.matmul(E, torch.linalg.pinv(F + I_fun(xx, hyperp, weights, biases)))
        )
        return float(val.detach().cpu())

    # Random grid search over the complementary coordinates
    # Criterion can be NaN (singular pinv) — keep a finite fallback candidate.
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

    # Assemble full design in original coordinate order
    out = torch.zeros(hyperp.I, dtype=dtype, device=device)
    out[ll_dist] = xd
    if n_exploit > 0:
        out[ll_exploit] = solution_grid
    return out
