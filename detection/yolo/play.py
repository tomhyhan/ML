import torch

x = torch.rand(3, 7, 7)

print(x[0].unsqueeze(0).shape)