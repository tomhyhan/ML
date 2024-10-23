import torch
from torch import nn
from functools import partial
from torchvision.ops.misc import Conv2dNormActivation

from maxvit_block2 import MaxVITBlock2
from utils2 import get_block_input_sizes, get_conv_size

class MaxVIT2(nn.Module):
    def __init__(
        self,
        input_img_size,
        stem_channels,
        block_channels,
        n_layers,
        partition_size,
        sqeeze_ratio,
        expension_ratio,
        mlp_ratio,
        head_dim,
        dropout,
        attention_dropout,
        stochatic_depth_p,
        num_classes
    ):
        norm_layer = partial(nn.BatchNorm2d, eps=1e-5, momentum=0.1)
        activation_layer = nn.GELU
        
        in_channel = 3
        
        block_input_sizes = get_block_input_sizes(input_img_size)
        for input_size in block_input_sizes:
            if input_size[0] % partition_size != 0 or input_size[1] % partition_size != 0:
                raise ValueError("Input size must be divisible by the partition size")
            
        self.stem = nn.Sequential(
            Conv2dNormActivation(
                in_channels=in_channel,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                activation_layer=activation_layer,
                norm_layer=norm_layer,
                inplace=None,
                bias=False
            ),
            Conv2dNormActivation(
                in_channels=stem_channels,
                out_channels=stem_channels,
                kernel_size=3,
                stride=1,
                activation_layer=None,
                norm_layer=None,
                inplace=None,
                bias=True
            )
        )
    
    
        input_size = get_conv_size(input_img_size) 
        in_channels = [stem_channels] + block_channels[:-1]
        out_channels = block_channels
        
        p_idx = 0
        stochatic_p = torch.linspace(0, stochatic_depth_p, steps=sum(n_layers), dtype=torch.Float32)
        
        self.layers = []
        for in_channel, out_channel, layers in zip(in_channels, out_channels, n_layers):
            self.layers.append(
                MaxVITBlock2(
                    in_channel=in_channel,
                    out_channel=out_channel,
                    n_layer=layers,
                    partition_size=partition_size,
                    head_dim=head_dim,
                    grid_size=input_size,
                    sqeeze_ratio=sqeeze_ratio,
                    expension_ratio=expension_ratio,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    stochatic_p=stochatic_p[p_idx:p_idx+layers]
                )    
            )
            p_idx += layers
            input_size = self.layers[-1].grid_size
            
        self.classcifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(block_channels[-1]),
            nn.Linear(block_channels[-1],block_channels[-1]),
            nn.Tanh(),
            nn.Linear(block_channels[-1], num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        x = self.classcifier(x)
        return x