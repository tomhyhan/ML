import torch
from torch import nn
from utils1 import get_conv_size
from maxvit_layer1 import MaxVITLayer1

class MaxVITBlock1(nn.Module):
    def __init__(
        self,
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
        grid_size,
        n_layers,
        stochastic_depths
    ):
        self.layers = nn.ModuleList()
        self.grid_size = get_conv_size(grid_size)
        
        for idx, p in enumerate(stochastic_depths):
            stride = 2 if idx == 0 else 1
            self.layers.append(
                MaxVITLayer1(
                    in_channel=in_channel if idx == 0 else out_channel,
                    out_channel=out_channel,
                    stride=stride,
                    head_dim=head_dim,
                    partition=partition,
                    squeeze_ratio=squeeze_ratio,
                    expension_ratio=expension_ratio,
                    activation_layer=activation_layer,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    grid_size=self.grid_size,
                    stochastic_depths_p=p
                )    
            )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x