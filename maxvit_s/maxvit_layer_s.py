from torch import nn
from collections import OrderedDict
from mbconv import MBConv
from partition_attn import PartitionAttentionLayer

class SimpleMaxVITLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        sqeeze_ratio,
        expansion_ratio,
        stride,
        norm_layer,
        activation_layer,
        head_dim,
        mlp_ratio,
        mlp_dropout,
        attention_dropout,
        partition_size,
        grid_size,
        p_stochastic_dropout
    ):
        super().__init__()
        layers = OrderedDict()
        
        layers["MBconv"] = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            sqeeze_ratio=sqeeze_ratio,
            stride=stride,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            p_stochastic_dropout=p_stochastic_dropout
        )
        
        layers["window_attention"] = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="window",
            grid_size=grid_size,
            mlp_ratio=mlp_ratio,
            mlp_dropout=mlp_dropout,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            attention_dropout=attention_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
        )
                
        layers["grid_attention"] = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="grid",
            grid_size=grid_size,
            mlp_ratio=mlp_ratio,
            mlp_dropout=mlp_dropout,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            attention_dropout=attention_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
        )        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
