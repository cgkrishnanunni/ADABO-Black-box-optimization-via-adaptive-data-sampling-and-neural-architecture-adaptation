"""
PyTorch ``DataLoader`` helpers that load the current labeled dataset from
``hyperp.data`` / ``hyperp.labels`` (written by ``ProposedOptimizer``).
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def get_data(batch_size, hyperp):
    """Return a training DataLoader for the surrogate SGD loop."""

    class Data(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.float32))
            self.len = self.X.shape[0]
       
        def __getitem__(self, index):
            return self.X[index], self.y[index]
   
        def __len__(self):
            return self.len
   
    batch_size = batch_size



# Instantiate training and test data
    train_data = Data(np.loadtxt(hyperp.data), np.loadtxt(hyperp.labels))
    if batch_size == "full_batch":
        train_dataloader = DataLoader(dataset=train_data, batch_size=len(np.loadtxt(hyperp.data)), shuffle=False)
    else:
        train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

    #test_data = Data(np.loadtxt("DATA/data_validation.data"), np.loadtxt("DATA/labels_validation.data"))
 
    #test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader
