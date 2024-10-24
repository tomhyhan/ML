import torch
from torch import nn
from maxvit_layer2 import MaxVITLayer2
from utils2 import get_conv_size

class MaxVITBlock2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers,
        partition_size,
        head_dim,
        norm_layer,
        activation_layer,
        grid_size,
        squeeze_ratio,
        expension_ratio,
        mlp_ratio,
        dropout,
        attention_dropout,
        stochatic_p
    ):
        
        self.grid_size = get_conv_size(grid_size)
        
        layers = []
        for idx, p in enumerate(stochatic_p):
            stride = 1 if idx == 0 else 2
            layers.append(
                MaxVITLayer2(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    stride=stride,
                    head_dim=head_dim,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    partition_size=partition_size,
                    grid_size=grid_size,
                    squeeze_ratio=squeeze_ratio,
                    expension_ratio=expension_ratio,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    stochatic_p=p,
                )
            )
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)