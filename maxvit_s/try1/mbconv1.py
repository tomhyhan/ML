import torch
from torch import nn
from collections import OrderedDict
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.ops.stochastic_depth import StochasticDepth
from sqeezeexc import SqeezeExciation

class MBConv1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        sqeeze_ratio,
        expension_ratio,
        activation_layer,
        norm_layer,
        stochastic_depths
    ):
        
        downsample = []
        self.downsample = []
        should_downsample = stride != 1 or in_channels != out_channels
        
        if should_downsample:
            if stride == 2:
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=3, stride=stride, 
                        padding=1
                    )
                )
            downsample.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )
            self.downsample = nn.Sequential(*downsample)
        else:
            self.downsample = nn.Identity()
        
        sqeeze_channels = int(in_channels * sqeeze_ratio)
        mid_channels = int(in_channels * expension_ratio)
        
        layers = OrderedDict()
        
        layers["pre_norm"] = norm_layer(in_channels)
        layers["1x1"] = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            in_place=None
        )  
        layers["3x3"] = Conv2dNormActivation(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            in_place=False
        )  
        layers["sqeeze_excitation"] = SqeezeExciation(
            input_channels=mid_channels,
            sqeeze_channels=sqeeze_channels,
            activation=nn.SiLU
        )
        layers=["1x1_2"] = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.layers = nn.Sequential(*layers)
        self.stochatic_depth = StochasticDepth(p=stochastic_depths, mode="row")
    
    def forward(self, x):
        downsample = self.downsample(x)
        return self.stochatic_depth(self.layers(x))