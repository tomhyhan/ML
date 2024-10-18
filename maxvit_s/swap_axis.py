import torch
from torch import nn

class SwapAxis(nn.Module):
    def __init__(self, a, b) -> None:
        super().__init__()
        self.a = a
        self.b = b
        
    def forward(self,x):
        x = torch.swapaxes(x, self.a, self.b)
        return x