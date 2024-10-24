import torch
from torch import nn
from collections import OrderedDict
from torchvision.ops.stochastic_depth import StochasticDepth

from mbconv2 import MBConv2
from partitionattn2 import PartitionAttn2

class MaxVITLayer2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        head_dim,
        norm_layer,
        activation_layer,
        partition_size,
        grid_size,
        squeeze_ratio,
        expension_ratio,
        mlp_ratio,
        dropout,
        attention_dropout,
        stochatic_p
    ):
        layers = OrderedDict()
        
        layers["MBConv"] = MBConv2(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            squeeze_ratio=squeeze_ratio,
            expension_ratio=expension_ratio,
        )
        layers["PartitionAttn"] = PartitionAttn2(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="window",
            grid_size=grid_size,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochatic_p=stochatic_p
        )
        layers["PartitionAttn"] = PartitionAttn2(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="grid",
            grid_size=grid_size,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochatic_p=stochatic_p
        )

        self.layers = nn.Sequential(*layers)
        self.stochatic_depth = StochasticDepth(p=stochatic_p, mode="row")
        
    def forward(self, x):
        x = self.stochatic_depth(self.layers(x))
        return x