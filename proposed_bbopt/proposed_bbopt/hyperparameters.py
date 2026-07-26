"""
Default hyperparameters for the proposed black-box optimizer.

Defaults match
``BLACK_BOX_OPTIMIZATION/.../PROPOSED_ALGORITHM/main.ipynb``
(stagnation-triggered AL, 4D Rosenbrock).

Most users only need to set:

- ``vec_lower`` / ``vec_upp`` (or pass ``bound=`` to ``ProposedOptimizer``)
- ``N_t_initial``, ``N_t_total`` (evaluation budget)
- ``P``, ``Tmax`` (Cuckoo Search effort on the surrogate)
- ``iter`` (surrogate training epochs per outer iteration)
- ``stagnation_m`` (Cuckoo non-improvements before one AL step)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np


@dataclass
class Hyperparameters:
    """
    Configuration for :class:`proposed_bbopt.ProposedOptimizer`.

    Design variables are optimized in normalized coordinates ``[0, 1]^n``
    and mapped to physical bounds via ``vec_lower`` / ``vec_upp``.
    Callers always pass / receive **physical** coordinates; normalization
    is handled inside the optimizer.
    """

    # ----- Experimental design / outer loop -----
    # Number of initial labeled designs when initialization=None.
    N_t_initial: int = 5
    # Total black-box evaluations (initial DoE + acquired points).
    N_t_total: int = 100
    # Number of successive non-improving *Cuckoo* iterations before one
    # active-learning (exploration) step. After that AL acquisition,
    # Cuckoo resumes and the counter resets.
    stagnation_m: int = 3

    # Physical box bounds — default: 4D Rosenbrock on [-2, 2]^4
    # (global minimum at x* = (1,1,1,1) with f(x*) = 0).
    vec_lower: np.ndarray = field(
        default_factory=lambda: np.array([-2.0, -2.0, -2.0, -2.0])
    )
    vec_upp: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 2.0, 2.0, 2.0])
    )
    # Normalized search box used internally (almost always [0, 1]).
    normalize: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))

    # ----- Active-learning internals -----
    # Monte Carlo samples when estimating the sensitivity Gram matrix.
    asymp_N: int = 200
    # Random grid searches inside active_learning.
    grid_opt: int = 100

    # ----- Cuckoo Search on the surrogate -----
    P: int = 200      # population (nests)
    Tmax: int = 500   # generations

    # ----- Surrogate architecture -----
    L: int = 4        # initial number of layers (grows if validation stalls)
    Nh: int = 20      # hidden units per layer
    sparsity: int = 10  # number of active neurons kept in a hidden layer
    # Structured sparsity on selected input→hidden connections at init.
    sparsity_input_Nh: np.ndarray = field(
        default_factory=lambda: np.array([0])
    )
    sparsity_input_I: np.ndarray = field(
        default_factory=lambda: np.array([1])
    )

    # ----- Surrogate training (SGD / AdamW) -----
    alpha: float = 0.01           # learning rate
    iter: int = 2000              # epochs per outer iteration
    back_track: int = 2000        # max steps when inserting a new layer
    batch_size: Union[str, int] = "full_batch"
    lr_decay: float = 1.0
    back_track_rate: float = 0.1
    std: float = 0.01             # weight init scale
    second_deriv: float = 1.0     # spectral criterion scale for layer insert

    # ----- Set / updated at runtime (usually leave alone) -----
    I: Optional[int] = None       # input dimension (= len(vec_lower))
    O: int = 1                    # output dimension (scalar loss)
    path: Optional[str] = None    # checkpoint path for best model
    data: Optional[str] = None    # path to training designs file
    labels: Optional[str] = None  # path to training labels file
    epsilon: float = 0.0          # layer-insertion step size (tuned by backtracking)

    def __post_init__(self) -> None:
        """Normalize array fields and infer input dimension ``I``."""
        self.vec_lower = np.asarray(self.vec_lower, dtype=float).reshape(-1)
        self.vec_upp = np.asarray(self.vec_upp, dtype=float).reshape(-1)
        if self.vec_lower.shape != self.vec_upp.shape:
            raise ValueError("vec_lower and vec_upp must have the same shape")
        self.I = int(self.vec_lower.shape[0])
        self.normalize = np.asarray(self.normalize, dtype=float).reshape(-1)
        self.sparsity_input_I = np.asarray(self.sparsity_input_I, dtype=int)
        self.sparsity_input_Nh = np.asarray(self.sparsity_input_Nh, dtype=int)

    @classmethod
    def from_bound(
        cls,
        bound: Sequence[tuple[float, float]],
        **kwargs,
    ) -> "Hyperparameters":
        """
        Build hyperparameters from a CSO-style ``bound=[(L,U), ...]`` list.

        Extra keyword arguments are forwarded to the constructor
        (e.g. ``N_t_total=100``).
        """
        lower = np.array([float(b[0]) for b in bound], dtype=float)
        upper = np.array([float(b[1]) for b in bound], dtype=float)
        return cls(vec_lower=lower, vec_upp=upper, **kwargs)
