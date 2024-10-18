from torch import nn
from collections import OrderedDict
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.ops import Conv2dNormActivation
from squeeze_exc import SqeezeExciation

class MBConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion_ratio,
        sqeeze_ratio,
        stride,
        activation_layer,
        norm_layer,
        p_stochastic_dropout
    ):
        downsamples = []
        self.downsample = []
        should_downsample = stride != 1 or in_channels != out_channels
        
        if should_downsample:
            if stride == 2:
                downsamples.append(nn.AvgPool2d(kernel_size=3, stride=stride, padding=1))
            downsamples.append(nn.Conv2d(in_channels, out_channels, 1, 1))
            self.downsample = nn.Sequential(*downsamples)            
        else:
            self.downsample = nn.Identity()
        
        mid_channels = int(out_channels * expansion_ratio)
        sqz_channels = int(out_channels * sqeeze_ratio)
        
        if p_stochastic_dropout:
            self.stochastic_depth = StochasticDepth(p=p_stochastic_dropout, mode="row")
        else:
            self.stochastic_depth = nn.Identity
            
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
            inplace=None
        ) 
        layers["conv_b"] = Conv2dNormActivation(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            inplace=None
        ) 
        layers["sqeeze_exciattion"] = SqeezeExciation(
            input_channels=mid_channels,
            squeeze_channels=sqz_channels,
            activation=nn.SiLU
        )
        layers=["conv_c"] = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.layers = nn.Sequential(layers)
            
    def forward(self, x):
        downsample = self.downsample(x)
        x = self.stochastic_depth(self.layers(x))
        return x + downsample