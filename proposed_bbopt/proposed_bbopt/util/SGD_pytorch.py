"""
Train the residual neural surrogate with PyTorch SGD.

Builds an ``nn.Module`` whose layers are initialized from the current
``weights`` / ``biases``, trains for ``hyperp.iter`` epochs on
``hyperp.data`` / ``hyperp.labels``, and returns updated parameters.
"""

import torch
import numpy as np
from torch import nn
from proposed_bbopt.util.get_data import get_data


def SGD_pytorch(
    hyperp, O, activation_input, activation, activation_output,
    weights, biases, I, batch_size, X_test, Y_test, Nt,
):
    """
    Fit the surrogate to the current labeled dataset.

    Returns
    -------
    loss_history, weights, biases, model
    """
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    input_dim = I
    hidden_dim = hyperp.Nh
    output_dim = O

    class NeuralNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, weights, biases, hyperp):
            super(NeuralNetwork, self).__init__()
            self.layer_0 = nn.Linear(input_dim, hidden_dim)
            self.layer_0.weight.data = weights[0].double()
            self.layer_0.bias.data = biases[0].double()

            # Same attributes as before (layer_1, layer_2, ...) without exec
            for i in range(1, hyperp.L - 2):
                layer = nn.Linear(hidden_dim, hidden_dim)
                layer.weight.data = weights[i].double()
                layer.bias.data = biases[i].double()
                setattr(self, f"layer_{i}", layer)

            self.layer_end = nn.Linear(hidden_dim, output_dim)
            self.layer_end.weight.data = weights[hyperp.L - 2].double()
            self.layer_end.bias.data = biases[hyperp.L - 2].double()

            self._hidden_layer_ids = list(range(1, hyperp.L - 2))

        def forward(self, x, hyperp):
            x = activation_input(self.layer_0(x))
            for i in self._hidden_layer_ids:
                layer = getattr(self, f"layer_{i}")
                x = x + activation(layer(x))
            x = activation_output(self.layer_end(x))
            return x

    model = NeuralNetwork(input_dim, hidden_dim, output_dim, weights, biases, hyperp)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperp.alpha)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyperp.lr_decay)

    num_epochs = hyperp.iter
    loss_values = []
    loss_test_values = []
    train_dataloader = get_data(batch_size, hyperp)

    # Precompute sparsity index grid (same pairs as nested io/jo loops)
    spars_rows = np.asarray(hyperp.sparsity_input_Nh)
    spars_cols = np.asarray(hyperp.sparsity_input_I)

    torch.save(model.state_dict(), hyperp.path)

    for epoch in range(num_epochs):
        for data in train_dataloader:
            X, y = data
            y = torch.reshape(y, (len(y), 1))
            X = X.double()
            y = y.double()

            # Zero structured connections (identical to double Python loop)
            model.layer_0.weight.data[np.ix_(spars_rows, spars_cols)] = 0

            optimizer.zero_grad()
            pred = model(X, hyperp)
            loss = loss_fn(pred, y)
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()

        scheduler.step()
        torch.save(model.state_dict(), hyperp.path)

    print("Training Complete")

    # Same tensors as torch.load(hyperp.path) after the final save above
    a = model.state_dict()

    torch.set_default_dtype(torch.float32)

    weights_new = [a["layer_0.weight"]]
    biases_new = [a["layer_0.bias"]]
    for i in range(1, hyperp.L - 2):
        weights_new.append(a[f"layer_{i}.weight"])
        biases_new.append(a[f"layer_{i}.bias"])
    weights_new.append(a["layer_end.weight"])
    biases_new.append(a["layer_end.bias"])

    del model
    del optimizer
    return loss_values, weights_new, biases_new, loss_test_values
