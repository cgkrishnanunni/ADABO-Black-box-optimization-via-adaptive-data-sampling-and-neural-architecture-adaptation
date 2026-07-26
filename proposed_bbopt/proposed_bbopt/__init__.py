"""
proposed_bbopt
==============

Surrogate-assisted black-box optimization with topological layer insertion.

Typical usage
-------------
1. Define a loss ``fitness(x) -> float`` in **physical** coordinates.
2. Provide box bounds ``[(L, U), ...]`` (one pair per design variable).
3. Optionally provide an initial DoE ``initialization`` of shape ``(N0, n)``.
4. Call ``ProposedOptimizer(...).execute()`` and read ``result.best``.

Example
-------
>>> from proposed_bbopt import ProposedOptimizer, Hyperparameters
>>> import numpy as np
>>> def rosenbrock(x):
...     x = np.asarray(x, dtype=float).reshape(-1)
...     return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))
>>> opt = ProposedOptimizer(
...     fitness=rosenbrock,
...     bound=[(-2.0, 2.0)] * 4,
...     N_t_total=100,
...     stagnation_m=3,
...     verbose=True,
... )
>>> result = opt.execute()
>>> result.best, result.best_fitness
"""

from proposed_bbopt.hyperparameters import Hyperparameters
from proposed_bbopt.optimizer import OptimizeResult, ProposedOptimizer

__all__ = ["Hyperparameters", "ProposedOptimizer", "OptimizeResult"]
__version__ = "0.1.0"
