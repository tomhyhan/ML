import torch
from torch import nn
from functools import partial
from torchvision.ops.misc import Conv2dNormActivation
from try1.maxvit_block1 import MaxVITBlock1

from utils1 import get_block_input_sizes, get_conv_size

class MaxVIT(nn.Module):
    def __init__(
        self,
        input_img_size,
        stem_channel,
        channels,
        layers,
        head_dim,
        partition,
        squeeze_ratio,
        expension_ratio,
        activation_layer,
        norm_layer,
        mlp_ratio,
        dropout,
        stochatic_depth_p,
        attention_dropout,
        num_classes
    ):
        super().__init__()
        
        in_channels = 3
        norm_layer = partial(nn.BatchNorm2d, eps=1e-5) 
        activation_layer = nn.GELU
        
        block_input_size = get_block_input_sizes(input_img_size)
        
        for block_input in block_input_size:
            if block_input[0] % partition != 0 or block_input[1] % partition != 0:
                raise ValueError("input size must de divisible by the partition size")
            
        self.stem = nn.Sequential(
            Conv2dNormActivation(
                in_channels,
                stem_channel,
                3,
                2,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                bias=False
            ),
            Conv2dNormActivation(
                stem_channel,
                stem_channel,
                3,
                1,
                norm_layer=None,
                activation_layer=None,
                bias=True
            )
        )
        
        self.blocks = nn.ModuleList()
        self.grid_size = get_conv_size(input_img_size)
        
        in_channels = [stem_channel] + channels[:-1]
        out_channels = channels
        
        stochastic_depths = torch.linspace(0, stochatic_depth_p, sum(layers)).tolist()
        p_id = 0
        
        for (in_channel, out_channel, n_layers) in zip(in_channel, out_channels, layers):
            self.blocks.append(
                MaxVITBlock1(
                    in_channel,
                    out_channel,
                    head_dim,
                    partition,
                    squeeze_ratio,
                    expension_ratio,
                    activation_layer,
                    norm_layer,
                    mlp_ratio,
                    dropout,
                    attention_dropout,
                    self.grid_size,
                    n_layers,
                    stochastic_depths[p_id:p_id+n_layers]
                )
            )
            self.grid_size = self.blocks[-1].grid_size
            p_id += n_layers
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], channels[-1]),
            nn.Tanh(),
            nn.Linear(channels[-1], num_classes, bias=False)
        )
        
        # init weights
        
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return x