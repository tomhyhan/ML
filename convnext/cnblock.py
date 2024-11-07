import torch
from torch import nn
from functools import partial
from torchvision.ops.misc import Permute
from torchvision.ops.stochastic_depth import StochasticDepth

class CNBlock(nn.Module):
    def __init__(
        self,
        dim, 
        layer_scale,
        sd_prob,
        norm_layer=None
    ):
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm(dim), eps=1e-6)
        # ResNeXt-ify (depthwise convolution)
        # Moving up depthwise conv layer
        # Increasing the kernel size
        # Replacing ReLU with GELU
        # Fewer activation functions
        # Fewer normalization layers
        # Substituting BN with LN
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0,2,3,1]),
            nn.Linear(in_features=dim, out_features=dim*4, bias=True),
            nn.GELU(),
            nn.Linear(in_features=dim*4, out_features=dim, bias=True),
            Permute([0,3,1,2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(sd_prob, mode="row")
        
        
    def forward(self, x):
        out = self.layer_scale * self.block(x)
        out = self.stochastic_depth(out)
        out += x
        return out 

class CNBlockConfig:
    def __init__(
        self,
        input_channels,
        output_channels,
        num_layers,
    ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_layers = num_layers
        
        