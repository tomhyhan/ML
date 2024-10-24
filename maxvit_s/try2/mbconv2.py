import torch
from torch import nn
from collections import OrderedDict
from torchvision.ops.misc import Conv2dNormActivation

from sqeezeexc2 import SqeezeExciation2

class MBConv2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        norm_layer,
        activation_layer,
        squeeze_ratio,
        expension_ratio
    ):
        should_downsample = stride != 2 or in_channels != out_channels
        downsample = []
        if should_downsample:
            if stride == 2:
                downsample.append(nn.AvgPool2d(
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                ))
            downsample.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                )
            )
            self.downsample = nn.Sequential(*downsample)
        else:
            self.downsample = nn.Identity()
        
        squeeze_channels = int(in_channels * squeeze_ratio)
        mid_channels = int(in_channels * expension_ratio)
        
        layers = OrderedDict() 
        layers["pre_norm"] = norm_layer(in_channels)
        layers["conv_a"] = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            inplace=False
        )  
        layers["conv_b"] = Conv2dNormActivation(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            inplace=False
        )
        layers["sqeezeexc"] = SqeezeExciation2(
            squeeze_channels=squeeze_channels,
            expension_channels=mid_channels
        )
        layers["conv_c"] = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.layers=nn.Sequential(layers)        
    
    def forward(self, x):
        downsample = self.downsample(x)
        x = self.layers(x)
        return x + downsample
