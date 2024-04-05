import torch as t

ten = t.randn(20, 16, 50, 32)
print(ten.view(-1).shape)