import torch

boxes1 = torch.rand(5, 4)

area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])


print(area1.shape)