"""
Proposed surrogate-assisted black-box optimizer (pip-facing API).

High-level loop (stagnation-triggered AL, matching PROPOSED_ALGORITHM)
----------------------------------------------------------------------
1. Label an initial design of experiments (user-supplied or random).
2. Train a neural surrogate on the labeled data.
3. If validation loss stalls, insert a residual layer (topology adapt).
4. Propose a new normalized design:
   - default: Cuckoo Search on the surrogate (exploitation)
   - if the best true loss does not improve for ``stagnation_m`` successive
     Cuckoo iterations, the next step uses low-rank active learning
     (exploration), then Cuckoo resumes
5. Evaluate the true fitness at that point and append to the dataset.
6. Repeat until ``N_t_total`` evaluations are reached.

Usage (physical coordinates throughout)::

    ProposedOptimizer(
        fitness=my_loss,          # f(x) -> float
        bound=[(-2, 2)] * 4,      # 4D Rosenbrock box
        initialization=X0,        # optional (N0, n) physical DoE
        N_t_total=100,
        stagnation_m=3,
        verbose=True,
    ).execute()
"""

from __future__ import annotations

import os
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed

from proposed_bbopt.hyperparameters import Hyperparameters
from proposed_bbopt.util.activation_prop import (
    activation,
    activation_derivative,
    activation_input,
    activation_input_derivative,
    activation_output,
    activation_output_derivative,
)
from proposed_bbopt.util.active_functions_lowrank import active_learning
from proposed_bbopt.util.back_prop import adjoint_prop, forward_prop
from proposed_bbopt.util.back_tracking import back_tracking
from proposed_bbopt.util.optimal_csa import cuckoo
from proposed_bbopt.util.random_initialization import get_new_weights, get_weights
from proposed_bbopt.util.SGD_pytorch import SGD_pytorch
from proposed_bbopt.util.spectral_decomp import spectrum

FitnessFn = Callable[[np.ndarray], float]
BoundType = Sequence[Tuple[float, float]]


@dataclass
class OptimizeResult:
    """
    Result of :meth:`ProposedOptimizer.execute`.

    Attributes
    ----------
    best : ndarray, shape (n,)
        Best physical design found.
    best_fitness : float
        Loss at ``best``.
    history_fitness : ndarray, shape (N_outer,)
        Best loss after each outer (acquisition) iteration.
    history_solution : ndarray, shape (N_outer, n)
        Best physical design after each outer iteration.
    X : ndarray, shape (N, n)
        All labeled designs in physical coordinates.
    y : ndarray, shape (N,)
        Corresponding loss values.
    """

    best: np.ndarray
    best_fitness: float
    history_fitness: np.ndarray
    history_solution: np.ndarray
    X: np.ndarray
    y: np.ndarray


class ProposedOptimizer:
    """
    Surrogate-assisted black-box optimizer (proposed algorithm).

    Parameters
    ----------
    fitness : callable
        Objective ``f(x) -> float`` for a single design ``x`` in **physical**
        coordinates (same contract as standalone CSO).
    bound : list of (L, U), optional
        Box constraints per variable. If omitted, taken from ``hyperp``.
    initialization : ndarray, shape (N0, n), optional
        Initial design of experiments in **physical** coordinates.
        If ``None``, ``N_t_initial`` uniform samples in the box are drawn.
    hyperp : Hyperparameters, optional
        Full hyperparameter object. Keyword overrides below take precedence
        when provided explicitly to ``__init__``.
    N_t_initial, N_t_total, P, Tmax, ... :
        Convenience overrides forwarded into ``hyperp``.
    verbose : bool
        If True, print outer-loop progress (training noise still suppressed).
    seed : int
        NumPy seed used when drawing a random initial DoE.
    workdir : str or Path, optional
        Directory for temporary train/checkpoint files. Defaults to a
        temporary directory cleaned up after ``execute``.
    """

    def __init__(
        self,
        fitness: FitnessFn,
        bound: Optional[BoundType] = None,
        initialization: Optional[np.ndarray] = None,
        hyperp: Optional[Hyperparameters] = None,
        N_t_initial: Optional[int] = None,
        N_t_total: Optional[int] = None,
        P: Optional[int] = None,
        Tmax: Optional[int] = None,
        stagnation_m: Optional[int] = None,
        L: Optional[int] = None,
        Nh: Optional[int] = None,
        alpha: Optional[float] = None,
        iter: Optional[int] = None,
        verbose: bool = False,
        seed: int = 0,
        workdir: Optional[Union[str, Path]] = None,
        min: bool = True,
    ):
        if not callable(fitness):
            raise TypeError("fitness must be a callable f(x) -> float")
        if not min:
            raise NotImplementedError("Only minimization (min=True) is supported")

        # Build or copy hyperparameters, then apply bound / keyword overrides.
        if hyperp is None:
            if bound is not None:
                hyperp = Hyperparameters.from_bound(bound)
            else:
                hyperp = Hyperparameters()
        else:
            hyperp = replace(hyperp)

        if bound is not None:
            hyperp.vec_lower = np.array([float(b[0]) for b in bound], dtype=float)
            hyperp.vec_upp = np.array([float(b[1]) for b in bound], dtype=float)
            hyperp.I = int(len(bound))

        # Explicit constructor kwargs win over values already on hyperp.
        overrides = {
            "N_t_initial": N_t_initial,
            "N_t_total": N_t_total,
            "P": P,
            "Tmax": Tmax,
            "stagnation_m": stagnation_m,
            "L": L,
            "Nh": Nh,
            "alpha": alpha,
            "iter": iter,
        }
        for key, val in overrides.items():
            if val is not None:
                setattr(hyperp, key, val)

        if hyperp.I is None:
            hyperp.I = int(hyperp.vec_lower.shape[0])
        if hyperp.N_t_total < hyperp.N_t_initial:
            raise ValueError("N_t_total must be >= N_t_initial")

        self.fitness = fitness
        self.hyperp = hyperp
        self.initialization = (
            None if initialization is None else np.asarray(initialization, dtype=float)
        )
        self.verbose = bool(verbose)
        self.seed = int(seed)
        self.workdir = Path(workdir) if workdir is not None else None

        self.n = int(hyperp.I)
        self.best: Optional[np.ndarray] = None
        self.best_fitness: Optional[float] = None
        self.result: Optional[OptimizeResult] = None

    # ----- coordinate helpers (physical <-> normalized [0, 1]^n) -----
    def _to_physical(self, x_norm: np.ndarray) -> np.ndarray:
        """Map a normalized design in [0, 1]^n to physical coordinates."""
        h = self.hyperp
        x_norm = np.asarray(x_norm, dtype=float).reshape(-1)
        return h.vec_lower + x_norm * (h.vec_upp - h.vec_lower)

    def _to_normalized(self, x_phys: np.ndarray) -> np.ndarray:
        """Map a physical design to normalized coordinates in [0, 1]^n."""
        h = self.hyperp
        x_phys = np.asarray(x_phys, dtype=float).reshape(-1)
        return (x_phys - h.vec_lower) / (h.vec_upp - h.vec_lower)

    def _eval_normalized(self, x_norm: np.ndarray) -> float:
        """Evaluate the user fitness at a normalized design."""
        return float(self.fitness(self._to_physical(x_norm)))

    # ----- public API -----
    def execute(self) -> OptimizeResult:
        """
        Run the proposed algorithm and return an :class:`OptimizeResult`.

        Also sets ``self.best`` and ``self.best_fitness`` (physical coords).
        Temporary train/checkpoint files go under ``workdir`` (or a temp dir
        that is deleted when this method returns).
        """
        h = self.hyperp
        own_tmpdir = None
        if self.workdir is None:
            own_tmpdir = tempfile.TemporaryDirectory(prefix="proposed_bbopt_")
            workdir = Path(own_tmpdir.name)
        else:
            workdir = Path(self.workdir)
            workdir.mkdir(parents=True, exist_ok=True)

        # Working files expected by the internal training / Cuckoo helpers.
        data_dir = workdir / "DATA"
        net_dir = data_dir / "NETWORKS"
        data_dir.mkdir(parents=True, exist_ok=True)
        net_dir.mkdir(parents=True, exist_ok=True)

        h.path = str(net_dir / "best_model_proposed.pt")
        h.data = str(data_dir / "data_train_proposed.data")
        h.labels = str(data_dir / "labels_train_proposed.data")

        old_cwd = os.getcwd()
        try:
            # Some util helpers read relative paths / CWD-based files.
            os.chdir(workdir)
            result = self._run(data_dir)
        finally:
            os.chdir(old_cwd)
            if own_tmpdir is not None:
                own_tmpdir.cleanup()

        self.result = result
        self.best = result.best
        self.best_fitness = result.best_fitness
        return result

    def _run(self, data_dir: Path) -> OptimizeResult:
        """Core outer loop: DoE → train → (adapt) → acquire → label → repeat."""
        h = self.hyperp
        I = int(h.I)
        O = int(h.O)
        n0 = int(h.N_t_initial)

        # ----- 1) Initial design of experiments -----
        # User array in physical coords, or uniform random samples in the box.
        if self.initialization is None:
            np.random.seed(self.seed)
            X_phys = np.random.rand(n0, I) * (h.vec_upp - h.vec_lower) + h.vec_lower
        else:
            X_phys = np.asarray(self.initialization, dtype=float)
            if X_phys.ndim == 1:
                X_phys = X_phys.reshape(1, -1)
            if X_phys.shape[1] != I:
                raise ValueError(
                    f"initialization must have shape (N, {I}), got {X_phys.shape}"
                )
            n0 = X_phys.shape[0]
            h.N_t_initial = n0  # custom DoE size overrides N_t_initial

        # Label in parallel; store normalized designs for the surrogate.
        X_norm = np.vstack([self._to_normalized(row) for row in X_phys])
        y = np.array(
            Parallel(n_jobs=-1)(delayed(self._eval_normalized)(row) for row in X_norm),
            dtype=float,
        )

        # Persist DoE so SGD / Cuckoo helpers can reload via hyperp.data/labels.
        np.savetxt(data_dir / "initialization.data", X_norm)
        np.savetxt(data_dir / "data_train_new.data", X_norm)
        np.savetxt(data_dir / "labels_train_new.data", y)
        np.savetxt(data_dir / "data_test.data", X_norm)
        np.savetxt(data_dir / "labels_test.data", y)
        np.savetxt(data_dir / "data_validation.data", X_norm)
        np.savetxt(data_dir / "labels_validation.data", y)
        np.savetxt(h.data, X_norm)
        np.savetxt(h.labels, y)

        # Torch layout used by forward_prop: features x samples → (I, Nt)
        X = torch.tensor(np.transpose(X_norm))
        Y = torch.tensor(np.transpose(y))
        X_test = torch.tensor(np.transpose(X_norm))
        Y_test = torch.tensor(np.transpose(y))
        X_validation = torch.tensor(np.transpose(X_norm))
        Y_validation = torch.tensor(np.transpose(y))
        Nt = n0

        # ----- 2) Sparse random weight initialization -----
        torch.manual_seed(0)
        weights, biases = get_weights(h, I, O)
        q_one = weights[1]
        q_zero = weights[0]
        q_three = biases[1]
        for io in range(0, h.Nh - h.sparsity):
            q_one[io, :] = 0
            q_three[io] = 0
        weights[1] = q_one
        biases[1] = q_three
        for io in range(0, np.shape(h.sparsity_input_Nh)[0]):
            for jo in range(0, np.shape(h.sparsity_input_I)[0]):
                q_zero[h.sparsity_input_Nh[io], h.sparsity_input_I[jo]] = 0
        weights[0] = q_zero

        loss_fn = nn.MSELoss()

        def torch_eval_loss(Nt_loc, X_loc, Y_loc, w, b):
            """MSE between surrogate predictions and labels (validation metric)."""
            _, output = forward_prop(
                h, Nt_loc, O, X_loc,
                activation_input, activation, activation_output,
                w, b,
            )
            pred = output.T.double()
            yy = Y_loc.reshape(-1, 1).double()
            return loss_fn(pred, yy).item()

        loss_validation: List[float] = []
        store_f: List[float] = []
        store_sol: List[np.ndarray] = []
        torch.manual_seed(0)
        iut = 0  # consecutive outer iters with non-improving validation loss
        new_data_train = X_norm.copy()
        new_labels_train = y.copy()

        # Stagnation-triggered AL: count successive non-improving Cuckoo steps
        stagnation = 0
        force_al = False
        best_so_far = float(np.min(new_labels_train))

        # ----- 3) Outer loop: one new labeled point per iteration -----
        n_outer = int(h.N_t_total - h.N_t_initial)
        for sample in range(n_outer):
            # Train / retrain the surrogate (stdout from SGD is suppressed).
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    loss, weights, biases, _ = SGD_pytorch(
                        h, O, activation_input, activation, activation_output,
                        weights, biases, I, h.batch_size,
                        X_validation, Y_validation, Nt,
                    )

            qq_val = torch_eval_loss(
                np.shape(X_validation)[1], X_validation, Y_validation, weights, biases
            )
            loss_validation.append(qq_val)

            # Track whether validation loss is stalling / getting worse.
            if len(loss_validation) >= 2 and h.L < 10:
                if (loss_validation[-2] - loss_validation[-1]) <= 0:
                    iut += 1
                else:
                    iut = 0

            # ----- 3a) Topology adapt: insert a residual layer if stalled -----
            if iut >= 2:
                iut = 0
                if self.verbose:
                    print("Adapting by adding a layer")
                with open(os.devnull, "w") as devnull:
                    with redirect_stdout(devnull), redirect_stderr(devnull):
                        X_state, output = forward_prop(
                            h, Nt, O, X,
                            activation_input, activation, activation_output,
                            weights, biases,
                        )
                        lambda_state, lambda_output = adjoint_prop(
                            h, Nt, O, Y,
                            activation_input_derivative, activation_derivative,
                            activation_output_derivative,
                            output, X_state, weights, biases,
                        )
                        eigen_values, eigen_vectors, neurons = spectrum(
                            h, I, Nt, lambda_state, X, X_state, O, lambda_output
                        )
                        index = int(np.argmax(eigen_values))
                        h.epsilon = 0
                        weights_new, biases_new = get_new_weights(
                            h, weights, biases, index, eigen_vectors
                        )
                        h.L = h.L + 1
                        weights, biases, _, _, _ = back_tracking(
                            h, Nt, X, weights, biases, index, eigen_vectors, O, Y,
                            weights_new, biases_new,
                        )
                if self.verbose:
                    print(
                        "Layer added at location:",
                        index,
                        "With number of Neurons:",
                        neurons[index],
                    )

            # ----- 3b) Acquisition: Cuckoo by default; AL after Cuckoo stagnation -----
            do_al = force_al
            force_al = False

            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    if do_al:
                        solution = active_learning(h, X, weights, biases, Nt)
                        inp = np.array(solution.numpy(), dtype=float)
                    else:
                        solution = cuckoo(
                            h, weights, biases, np.loadtxt(h.data)
                        )
                        inp = np.asarray(solution, dtype=float).reshape(-1)

            if self.verbose:
                if do_al:
                    print(
                        f"Acquisition: active learning "
                        f"(Cuckoo stagnated for {h.stagnation_m} steps)"
                    )
                else:
                    print("Data acquisition: By minimizing surrogate")

            # ----- 3c) True black-box evaluation + append to dataset -----
            steady = self._eval_normalized(inp)

            bb = np.loadtxt(h.data)
            ll = np.loadtxt(h.labels)
            if bb.ndim == 1:
                bb = bb.reshape(1, -1)
            new_data_train = np.zeros((bb.shape[0] + 1, I))
            new_labels_train = np.zeros(bb.shape[0] + 1)
            new_data_train[:-1, :] = bb
            new_data_train[-1, :] = inp
            new_labels_train[:-1] = ll
            new_labels_train[-1] = steady
            np.savetxt(h.data, new_data_train)
            np.savetxt(h.labels, new_labels_train)
            X = torch.tensor(np.transpose(new_data_train))
            Y = torch.tensor(np.transpose(new_labels_train))
            Nt = Nt + 1

            y_min = float(np.min(new_labels_train))
            inde = int(np.argmin(new_labels_train))
            store_f.append(y_min)
            store_sol.append(new_data_train[inde].copy())

            # ----- 3d) Stagnation bookkeeping (only Cuckoo non-improvements count) -----
            if y_min < best_so_far:
                best_so_far = float(y_min)
                stagnation = 0
                if self.verbose:
                    print("Best improved → stagnation reset to 0")
            elif do_al:
                stagnation = 0
                if self.verbose:
                    print("AL step complete → resume Cuckoo, stagnation reset to 0")
            else:
                stagnation += 1
                if self.verbose:
                    print(
                        f"no improvement after current minimization (exploitation) → "
                        f"stagnation = {stagnation}/{h.stagnation_m}"
                    )
                if stagnation >= h.stagnation_m:
                    force_al = True
                    stagnation = 0
                    if self.verbose:
                        print("Adding data point with statistical active learning theory for exploration")

            if self.verbose:
                best_phys = self._to_physical(new_data_train[inde])
                print(f"iteration number: {sample}")
                print(f"Current minimum: {y_min}")
                print(f"best solution:{best_phys}")
                print(f"total function evaluations: {len(new_labels_train)}")

        # ----- 4) Package result in physical coordinates -----
        hist_f = np.asarray(store_f, dtype=float)
        hist_s_norm = np.asarray(store_sol, dtype=float)
        if hist_s_norm.ndim == 1:
            hist_s_norm = hist_s_norm.reshape(1, -1)
        hist_s = np.vstack([self._to_physical(row) for row in hist_s_norm])

        X_phys_all = np.vstack([self._to_physical(row) for row in new_data_train])
        best_idx = int(np.argmin(new_labels_train))
        best = X_phys_all[best_idx].copy()
        best_f = float(new_labels_train[best_idx])

        return OptimizeResult(
            best=best,
            best_fitness=best_f,
            history_fitness=hist_f,
            history_solution=hist_s,
            X=X_phys_all,
            y=new_labels_train.copy(),
        )
