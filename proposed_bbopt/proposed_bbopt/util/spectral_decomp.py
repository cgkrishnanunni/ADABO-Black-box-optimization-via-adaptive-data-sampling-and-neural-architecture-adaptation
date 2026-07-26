"""
Spectral criterion that decides *where* to insert a new residual layer
when surrogate validation loss stalls.
"""

import torch
import numpy as np
from numpy import linalg as LA


def spectrum(hyperp, I, Nt, lambda_state, X, X_state, O, lambda_output):
    """
    Spectral criterion for residual-layer insertion.

    For each candidate hidden layer, builds a block-diagonal matrix whose r-th
    block is
        b_r = second_deriv * sum_s λ_{s,r} x_s x_s^T
    then returns a cumulative leading-eigenvalue score and the summed
    eigenvectors used to initialize the new layer.
    """
    torch.manual_seed(0)
    eigen_values = []
    eigen_vectors = []
    neurons = []
    Nh = hyperp.Nh
    N = Nh * Nh
    sd = hyperp.second_deriv

    for lay in range(1, hyperp.L - 1):
        # Convert once per layer (same values as per-sample .numpy() calls)
        lam_all = np.nan_to_num(
            lambda_state[lay - 1].detach().cpu().numpy(),
            nan=0.0, posinf=0.0, neginf=0.0,
        )  # (Nt, Nh)
        xs = np.nan_to_num(
            X_state[lay - 1].detach().cpu().numpy(),
            nan=0.0, posinf=0.0, neginf=0.0,
        )  # (Nt, Nh)

        # Block-diagonal Hessian proxy: block r = sd * X^T diag(λ_{:,r}) X
        a = np.zeros((N, N))
        for r in range(Nh):
            # Identical to sum_s lam[s,r] * outer(xs[s], xs[s]) * sd
            b = sd * (xs.T * lam_all[:, r]) @ xs
            a[r * Nh:(r + 1) * Nh, r * Nh:(r + 1) * Nh] = b

        if np.isfinite(a).all():
            values, vectors = LA.eigh(a)
        else:
            values, vectors = LA.eigh(np.zeros((N, N)))
            print('bad vector')

        # Ascending sort; inds[-1], inds[-2], ... are largest eigenvalues
        inds = np.argsort(values)

        tutu = 0
        tutu_vec = 0 * vectors[:, inds[N - 1]]
        n_selected = Nh
        tutu_best = None

        # Same loop as before: iu = 1 .. Nh-1 over largest eigenvalues
        for iu in range(1, Nh):
            idx = inds[N - iu]
            tutu = tutu + values[idx]
            tutu_vec = tutu_vec + vectors[:, idx]

            if iu == 1:
                tutu_best = tutu

            if np.isfinite(tutu_best) and abs(tutu_best) > 1e-16:
                if abs(values[idx] - tutu_best) / abs(tutu_best) > 0.5:
                    n_selected = iu
                    break

        neurons.append(n_selected)
        eigen_values.append(tutu)
        eigen_vectors.append(tutu_vec)

    return eigen_values, eigen_vectors, neurons
