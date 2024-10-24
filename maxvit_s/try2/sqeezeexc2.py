import torch
from torch import nn

class SqeezeExciation2(nn.Module):
    def __init__(
        self,
        squeeze_channels,
        expension_channels
    ):
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(expension_channels, squeeze_channels, kernel_size=1)
        self.activate = nn.SiLU()
        self.fc2 = nn.Conv2d(squeeze_channels, expension_channels,kernel_size=1)
        self.scale_activation = nn.Sigmoid()
    
    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.activate(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale
        