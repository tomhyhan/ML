import torch 
from torch import nn


x= torch.randn(1, 10, 4, 4)
m = nn.AdaptiveAvgPool2d(1)

print(m(x).shape)