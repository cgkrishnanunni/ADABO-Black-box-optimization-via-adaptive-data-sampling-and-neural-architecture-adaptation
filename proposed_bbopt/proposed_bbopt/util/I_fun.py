"""
Sensitivity Gram matrix ``I(x)`` for active-learning acquisition.

``I(x) = J(x)^T J(x)`` where ``J`` is the Jacobian of the network output
w.r.t. the concatenated weight/bias parameter vector.
"""

import torch
from proposed_bbopt.util.activation_prop import activation, activation_input, activation_output


def I_fun(x, hyperp, weights, biases):
    """
    Sensitivity Gram matrix I(x) = J(x)^T J(x), where J is the Jacobian of
    the network output w.r.t. the concatenated weight/bias parameter vector.
    """
    # Match trained weight dtype/device (SGD uses float64; raw inputs may be float32)
    dtype = weights[0].dtype
    device = weights[0].device
    x = torch.as_tensor(x, dtype=dtype, device=device).reshape(-1)

    # Precompute flat sizes / slices (same packing order as the original loops)
    w_sizes = [int(w.numel()) for w in weights]
    b_sizes = [int(b.numel()) for b in biases]
    n_params = sum(w_sizes) + sum(b_sizes)

    new = torch.zeros(n_params, dtype=dtype, device=device)

    w_slices = []
    k = 0
    for i, w in enumerate(weights):
        k_old = k
        k = k + w_sizes[i]
        new[k_old:k] = w.reshape(-1)
        w_slices.append((k_old, k, w.shape))

    bias_offset = k  # start index of the bias block in `new`
    b_slices = []
    k_biases = bias_offset
    for i, b in enumerate(biases):
        k_old_biases = k_biases
        k_biases = k_biases + b_sizes[i]
        new[k_old_biases:k_biases] = b.reshape(-1)
        b_slices.append((k_old_biases, k_biases))

    def net(neww):
        # Unpack with fixed slices (equivalent to resetting the old globals each call)
        y = None
        for i in range(0, hyperp.L - 2):
            k_n_old, k_n, shape = w_slices[i]
            k_old_biases_n, k_biases_n = b_slices[i]
            W = neww[k_n_old:k_n].reshape(shape)
            bvec = neww[k_old_biases_n:k_biases_n]
            if i == 0:
                y = activation_input(torch.matmul(W, x) + bvec)
            else:
                y = y + activation(torch.matmul(W, y) + bvec)

        k_n_old, k_n, shape = w_slices[hyperp.L - 2]
        k_old_biases_n, k_biases_n = b_slices[hyperp.L - 2]
        W = neww[k_n_old:k_n].reshape(shape)
        bvec = neww[k_old_biases_n:k_biases_n]
        y = activation_output(torch.matmul(W, y) + bvec)
        return y

    kut = torch.autograd.functional.jacobian(net, new)
    # Same as torch.matmul(np.transpose(kut), kut) for real Jacobians
    return torch.matmul(kut.T, kut)
