"""
Cuckoo Search acquisition on the trained surrogate.

``cuckoo(...)`` minimizes the neural surrogate over the normalized box and
returns the best nest (next candidate design in ``[0, 1]^n``).
"""

import numpy as np
import torch

from proposed_bbopt.util.activation_prop import activation, activation_input, activation_output
from proposed_bbopt.util.back_prop import forward_prop
from proposed_bbopt.util.cso_modified import CSO


def fitness_1(X, hyperp, weights, biases):
    """
    Surrogate fitness for a CSO population.

    Parameters
    ----------
    X : array, shape (P, I)
        Population of nests in normalized coordinates (as passed by CSO).

    Returns
    -------
    output : ndarray, shape (P,)
        Surrogate predictions for each nest (minimized by CSO).
    """
    # forward_prop expects features x samples: (I, Nt)
    X_t = np.transpose(np.asarray(X, dtype=float))
    _, output = forward_prop(
        hyperp,
        X_t.shape[1],
        hyperp.O,
        X_t,
        activation_input,
        activation,
        activation_output,
        weights,
        biases,
    )
    out = output.detach().cpu().numpy().reshape(-1) if torch.is_tensor(output) else np.asarray(output).reshape(-1)
    return out.astype(float)


def cuckoo(hyperp, weights, biases, X, seed=0):
    """
    Run Cuckoo Search on the neural surrogate and return the best nest found.

    Search is in normalized coordinates [normalize[0], normalize[1]]^I.
    """
    np.random.seed(seed)
    initialization = np.random.rand(hyperp.P, hyperp.I)

    lo = float(hyperp.normalize[0])
    hi = float(hyperp.normalize[1])
    # Same per-variable box used before (only the first n=I entries are applied)
    bound = [(lo, hi) for _ in range(hyperp.I)]

    solution = CSO(
        fitness=fitness_1,
        data=X,
        weights=weights,
        biases=biases,
        hyperp=hyperp,
        Tmax=hyperp.Tmax,
        initialization=initialization,
        P=hyperp.P,
        n=hyperp.I,
        verbose=True,
        bound=bound,
        seed=seed,
    ).execute()

    return solution
