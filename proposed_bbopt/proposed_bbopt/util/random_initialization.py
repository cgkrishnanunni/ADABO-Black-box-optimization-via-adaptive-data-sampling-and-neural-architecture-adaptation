"""
Random weight initialization and construction of weights for a newly
inserted residual layer (used during topological adaptation).
"""

import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_weights(hyperp, I, O):
    """
    Draw initial ``weights`` / ``biases`` lists for an ``L``-layer residual net.

    Returns
    -------
    weights, biases : list of tensors
    """
    weights = []  # W for each layer
    biases = []   # b for each layer

    for i in range(0,hyperp.L-2):
    
        if i==0:
            weights.append(hyperp.std*torch.randn((hyperp.Nh,I)))
            biases.append(hyperp.std*torch.randn((hyperp.Nh,)))
           


        if i>0:
            weights.append(hyperp.std*torch.randn((hyperp.Nh,hyperp.Nh)))
            biases.append(hyperp.std*torch.randn((hyperp.Nh,)))



    weights.append(hyperp.std*torch.randn((O,hyperp.Nh)))
    biases.append(hyperp.std*torch.randn((O,)))




    return weights, biases



def get_new_weights(hyperp,weights,biases,index,eigen_vectors):
    weights_updated=[]
    biases_updated=[]
    
    for i in range(0,hyperp.L):
        if i<index+1:
            weights_updated.append(weights[i])
            biases_updated.append(biases[i])
        
        if i==index+1:
            # Match dtype of existing trained weights (often float64 after SGD)
            dtype = weights[0].dtype if torch.is_tensor(weights[0]) else torch.float32
            eigen_matrix = np.real(
                np.reshape(eigen_vectors[index], (hyperp.Nh, hyperp.Nh))
            )
            weights_updated.append(
                hyperp.epsilon * torch.tensor(eigen_matrix, dtype=dtype)
            )
            biases_updated.append(hyperp.epsilon * torch.zeros((hyperp.Nh,), dtype=dtype))

        if i>index+1:
            weights_updated.append(weights[i-1])
            biases_updated.append(biases[i-1])


    return weights_updated, biases_updated