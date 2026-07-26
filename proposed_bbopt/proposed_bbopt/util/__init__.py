"""
Internal surrogate / acquisition utilities for ``proposed_bbopt``.

These modules support :class:`proposed_bbopt.ProposedOptimizer` and are not
part of the stable public API. Prefer importing from ``proposed_bbopt`` itself.

| Module | Role |
|--------|------|
| ``activation_prop`` | Network activations and their derivatives |
| ``back_prop`` | Forward / adjoint (backprop) through the residual net |
| ``SGD_pytorch`` | Surrogate training with PyTorch SGD |
| ``random_initialization`` | Weight init and new-layer weight construction |
| ``spectral_decomp`` | Spectral criterion for where to insert a layer |
| ``back_tracking`` | Line-search when accepting a newly inserted layer |
| ``optimal_csa`` / ``cso_modified`` | Cuckoo Search on the surrogate |
| ``active_functions_lowrank`` | Default AL acquisition (low-rank A-opt; used by optimizer) |
| ``active_functions`` | Dense Gram AL (reference / alternative) |
| ``I_fun`` | Dense sensitivity Gram matrix (used by dense AL) |
| ``get_data`` | DataLoader helpers for surrogate training |
"""
