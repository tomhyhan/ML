# import torch
# from torch import nn
# from torchvision.ops.misc import Conv2dNormActivation

# print((224 - 3 + 2 * 0) // 2 + 1)
# print((111 - 3 + 2 * 0) // 1 + 1)
stride = 2
in_channels = 10
out_channels = 10

should_proj = stride != 1 or in_channels != out_channels
print(should_proj)