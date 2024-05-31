import torch

x = torch.tensor([1,2,3])

idn = torch.nn.Identity()

y = idn(x)
print(y)