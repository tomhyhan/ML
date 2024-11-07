import torch
from torch import nn
from torchvision.ops.misc import Permute
from torchvision.ops.stochastic_depth import StochasticDepth

class CNBlock1(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale,
        sd_prob
    ):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, padding=3, groups=dim),
            Permute([0,2,3,1]),
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=dim*4),
            nn.GELU(),
            nn.Linear(in_features=dim*4, out_features=dim),
            Permute([0,3,1,2])
        )
        
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(sd_prob, mode="row")
        
    def forward(self, x):
        out = self.block(x) * self.layer_scale
        out = self.stochastic_depth(out)
        out += x
        return out
        
class CNBlockSetting1:
    def __init__(
        self,
        input_channels,
        output_channels,
        num_layers
    ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_layers = num_layers   