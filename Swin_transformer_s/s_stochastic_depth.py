import torch
from torch import nn

def stochastic_depth(input, p, mode, training=True):
    if p < 0.0 or p > 1.0:
        raise ValueError("probability must be between 0 and 1")
    if mode not in ["row", "batch"]:
        raise ValueError("mode must be either 'batch' or 'row'")
    if p == 0.0 or not training:
        return input
    
    survival_rate = 1 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
        
    noise = torch.empty(size, input.device, input.dtype)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise

class SimpleStochasticDepths(nn.Module):
    
    def __init__(self, p, mode):
        self.p = p
        self.mode = mode
    
    def forward(self, x):
        return stochastic_depth(x, self.p, self.mode, self.training)