import torch

image = torch.rand(2, 3, 1200, 600)

scale = 16/600
print(float(torch.tensor(scale).log2().round()))
scale = 2 ** float(torch.tensor(scale).log2().round())
print("scale", scale)
