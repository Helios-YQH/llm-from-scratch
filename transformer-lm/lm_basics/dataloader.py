import numpy as np
import torch

def run_get_batch(dataset, batch_size, context_length, device):
    n = len(dataset) - context_length
    ix = torch.randint(n, (batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(dataset[i:i+context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(dataset[i+1:i+1+context_length].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)