import torch

x = torch.tensor([0,1])

y = torch.tensor([[5],[10]])

print(x.shape, y.shape)
print(x * y)

mask = torch.tensor([1, 0])        # Shape: (2,)
errors = torch.tensor([[0.25], [0.16]])  # Shape: (2, 1)
result = mask * errors
print(result)