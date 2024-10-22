import torch
from torch import nn
from collections import OrderedDict
from mbconv1 import MBConv1
from partitionattn1 import PartitionAttnLayer1

class MaxVITLayer1(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        stride,
        head_dim,
        partition,
        sqeeze_ratio,
        expension_ratio,
        activation_layer,
        norm_layer,
        mlp_ratio,
        dropout,
        attention_dropout,
        grid_size,
        stochastic_depths
    ):
        super().__init__()
        
        layers = OrderedDict()
        
        layers["MBConv"] = MBConv1(
            in_channels=in_channel,
            out_channels=out_channel,
            stride=stride,
            sqeeze_ratio=sqeeze_ratio,
            expension_ratio=expension_ratio,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            stochastic_depths=stochastic_depths
        )  
        
        layers["windowattn"] = PartitionAttnLayer1(
            in_channel=out_channel,
            head_dim=head_dim,
            partition=partition,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            grid_size=grid_size,
            mode="window",
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depths=stochastic_depths
        ) 
        
        layers["gridattn"] = PartitionAttnLayer1(
            in_channel=out_channel,
            head_dim=head_dim,
            partition=partition,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            grid_size=grid_size,
            mode="grid",
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depths=stochastic_depths
        ) 
        
        self.layers =nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layers(x)
        return x