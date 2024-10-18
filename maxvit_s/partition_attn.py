from torch import nn
from window_partition import WindowDepartition, WindowPartition
from swap_axis import SwapAxis
from relativeposattn import RelativePositionalMultiHeadAttn
from torchvision.ops.misc import MLP
from torchvision.ops.stochastic_depth import StochasticDepth

class PartitionAttentionLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        head_dim,
        partition_size,
        partition_type,
        grid_size,
        mlp_ratio,
        mlp_dropout,
        activation_layer,
        norm_layer,
        attention_dropout,
        p_stochastic_dropout
    ):
        super().__init__()
        
        self.n_heads = in_channels // head_dim
        self.head_dim = head_dim
        self.n_partitions = grid_size[0] // partition_size
        self.partition_type= partition_type
        self.grid_size = grid_size
        
        if partition_type not in ["grid", "window"]:
            raise ValueError("partition must be either 'grid' or ''window")
        
        if partition_type == "window":
            self.p, self.g = partition_size, self.n_partitions
        else:
            self.p, self.g = self.n_partitions, partition_size
            
        self.partition_op = WindowPartition()
        self.departition_op = WindowDepartition()
        self.partition_swap = SwapAxis(-2,-3) if partition_size == "grid" else nn.Identity()
        self.departition_swap = SwapAxis(-2,-3) if partition_size == "grid" else nn.Identity()
        
        self.attn_layer = nn.Sequential(
            norm_layer(in_channels),
            RelativePositionalMultiHeadAttn(
                in_channels,
                head_dim,
                partition_size**2
            ),
            nn.Dropout(p=attention_dropout)
        )
        
        self.norm = norm_layer(in_channels)
        self.mlp = MLP(in_channels, [in_channels*mlp_ratio, in_channels], dropout=mlp_dropout,  activation_layer=activation_layer)
        
        self.stochastic_dropout = StochasticDepth(p=p_stochastic_dropout, mode="row") 
        
    def forward(self, x):
        hp, wp = self.grid_size[0] // self.p, self.grid_size[1] // self.p
        
        x = self.partition_op(x, self.p)
        x = self.partition_swap(x)
        x = self.stochastic_dropout(self.attn_layer(x))
        x = self.stochastic_dropout(self.mlp(self.norm(x)))
        x = self.departition_swap(x)
        x = self.departition_op(x, hp, wp)
        return x
