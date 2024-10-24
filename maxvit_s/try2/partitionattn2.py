import torch
from torch import nn
from torchvision.ops.misc import MLP
from torchvision.ops.stochastic_depth import StochasticDepth

from relativeposattn2 import RelativePositionalMultiHeadAttn2

class PartitionAttn2(nn.Module):
    def __init__(
        self,
        in_channels,
        head_dim,
        partition_size,
        partition_type,
        grid_size,
        activation_layer,
        norm_layer,
        mlp_ratio,
        dropout,
        attention_dropout,
        stochatic_p
    ):
        self.n_heads = in_channels // partition_size
        self.head_dim = head_dim
        self.n_partition = grid_size[0] // partition_size
        
        self.partition_op = WindowPartition()
        self.departition_op = WindowDepartition()
        self.partition_swap = SwapAxes(-2,-3) if partition_type == "grid" else nn.Identity()
        self.departition_swap = SwapAxes(-2,-3) if partition_type == "grid" else nn.Identity()

        if partition_type == "window":
            self.p, self.g = partition_size, self.n_partition
        else:
            self.p, self.g = self.n_partition, partition_size
        
        self.norm1 = norm_layer(in_channels)
        self.attn_layer = RelativePositionalMultiHeadAttn(
            in_channels=in_channels,
            head_dim=head_dim,
            max_seq_len=partition_size**2
        )
        self.attn_dropout = nn.Dropout(p=attention_dropout)
        
        self.norm2 = norm_layer(in_channels)
        self.mlp = MLP(in_channels, [in_channels*mlp_ratio, in_channels], activation_layer=activation_layer, dropout=dropout)        
        
        self.stochatic_depth = StochasticDepth(p=stochatic_p, mode="row")
        
    def forward(self, x):
        B, C, H, W, = x.shape
        
        hp = H // self.p
        wp = W // self.p 
        
        x = self.partition_op(x, self.p)
        x = self.partition_swap(x)
        x = x + self.stochatic_depth(self.attn_dropout(self.attn_layer(self.norm1(x))))
        x = x + self.stochatic_depth(self.mlp(self.norm2(x)))
        x = self.departition_swap(x)
        x = self.departition_op(x, hp, wp, self.p)

        return x
    
class WindowPartition(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, p):
        B, C, H, W = x.shape
        
        num_windows = (H//p) * (W//p)
        x = x.reshape(B, C, H//p, p, W//p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, num_windows, p*p, C)
        return x
    
class WindowDepartition(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, hp, wp, p):
        B, hpwp, pp, C = x.shape
        
        x = x.reshape(B, hp, wp, p, p, C)
        x = x.permute(0,5,1,3,2,4)
        x = x.reshape(B, C, hp*p, wp*p)
        return x
    
class SwapAxes(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        
    def forward(self, x):
        x = torch.swapaxes(x, self.a, self.b)
        return x
    
