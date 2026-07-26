"""
Activation functions (and derivatives) used by the neural surrogate.

- ``activation_input``: first hidden layer
- ``activation``: residual / hidden layers
- ``activation_output``: output layer

Matching ``*_derivative`` helpers are used by adjoint / spectral layer insertion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _as_numpy(input):
    """Convert tensor or array-like to a NumPy float array."""
    if torch.is_tensor(input):
        return input.detach().cpu().numpy().astype(float, copy=False)
    return np.asarray(input, dtype=float)


def activation(input):
    """Hidden-layer activation: 2 * SiLU(x) - tanh(x)."""
    # Alternative activations kept for reference / experiments:
    # return 0.5 * (5 * input ** 3 - 3 * input)
    # return 2*input*(1/(1+np.exp(-input))) - tanh(input)
    return 2 * torch.nn.functional.silu(input) - torch.nn.functional.tanh(input)


def activation_input(input):
    """First-layer activation (same family as hidden; see source history for alts)."""
    # return (np.exp(input)-np.exp(-input))/(np.exp(input)+np.exp(-input))
    return torch.nn.functional.tanh(input)

    



def activation_output(input):

    
    #return 0.5 * (5 * input ** 3 - 3 * input)
   # return torch.nn.functional.softmax(input,dim=0)
    return input
    #return torch.nn.functional.sigmoid(input)

def activation_output_derivative(input):


  #  jacobian_m = np.diag(input)
   # for i in range(len(jacobian_m)):
   #     for j in range(len(jacobian_m)):
    #        if i == j:
     #           jacobian_m[i][j] = input[i] * (1-input[i])
    #        else: 
     #           jacobian_m[i][j] = -input[i] * input[j]
   # return jacobian_m



    #return torch.nn.functional.softmax(input,dim=1)*(1-np.exp(input))
    #return torch.nn.functional.sigmoid(input)*(1-torch.nn.functional.sigmoid(input))
    return 1

def activation_input_derivative(input):
    # d/dx tanh(x) = 1 - tanh(x)^2  (numerically stable; no raw exp)
    x = _as_numpy(input)
    t = np.tanh(x)
    return 1.0 - t * t


  

def activation_derivative(input):
    # Derivative of 2*silu(x) - tanh(x), with clipped sigmoid for stability.
    # silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    x = _as_numpy(input)
    x_clip = np.clip(x, -60.0, 60.0)
    sigmoid_comp = 1.0 / (1.0 + np.exp(-x_clip))
    silu_deriv = sigmoid_comp + x_clip * sigmoid_comp * (1.0 - sigmoid_comp)
    tan_comp = 1.0 - np.tanh(x) ** 2
    return 2.0 * silu_deriv - tan_comp
