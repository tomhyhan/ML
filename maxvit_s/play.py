import torch
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation

# print((224 - 3 + 2 * 0) // 2 + 1)
# print((111 - 3 + 2 * 0) // 1 + 1)

x = torch.rand(10,3,224,224)
conv = Conv2dNormActivation(
                3,
                64,
                3,
                stride=2,
                norm_layer = nn.BatchNorm2d,
                activation_layer=nn.GELU,
                bias=False,
                inplace=None,
)
conv2 = Conv2dNormActivation(
                64, 64, 3, stride=1, norm_layer=None, activation_layer=None,  bias=True
            )
r = conv(x)
print(r.shape)
# print(type(r))
# print(conv2(r).shape)

conv3 = nn.Conv2d(3,64,3,2)
print(conv3(x).shape)