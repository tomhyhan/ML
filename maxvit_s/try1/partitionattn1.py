import torch
from torch import nn
from torchvision.ops.misc import MLP
from torchvision.ops.stochastic_depth import StochasticDepth
import torch.nn.functional as F


class PartitionAttnLayer1(nn.Module):
    def __init__(
        self,
        in_channel,
        head_dim,
        partition_size,
        activation_layer,
        norm_layer,
        grid_size,
        mode,
        mlp_ratio,
        dropout,
        attention_dropout,
        stochastic_p
    ):
        self.n_heads = in_channel // head_dim
        self.head_dim = head_dim
        self.n_partitions = grid_size[0] // partition_size
        self.grid_size = grid_size
        self.mode = mode
        
        if mode not in ["window", "grid"]:
            raise ValueError("mode should be either window or grid")

        if mode == "window":
            self.p, self.g = partition_size, self.n_partitions
        else:
            self.p, self.g = self.n_partitions, partition_size
        
        self.partition_op = WindowPartition()
        self.departition_op = WindowDepartition()
        self.partition_swap = SwapAxis(-2,-3) if mode == "grid" else nn.Identity()
        self.departition_swap = SwapAxis(-2,-3) if mode == "grid" else nn.Identity()
        
        self.attn_layer = nn.Sequential(
            norm_layer(in_channel),
            RelativePositionalMultiHeadAttn(
                in_channels=in_channel,
                head_dim=head_dim,
                max_seq_len=partition_size**2
            ),
            nn.Dropout(p=attention_dropout)
        )
        
        self.norm = norm_layer(in_channel)
        self.mlp = MLP(in_channel, [in_channel*mlp_ratio, in_channel], dropout=dropout, activation_layer=activation_layer)

        self.stochastic_depth = StochasticDepth(p=stochastic_p, mode="row")
            
    def forward(self, x):
        
        hp, wp = self.grid_size[0] // self.p, self.grid_size[1] // self.p
        
        x = self.partition_op(x, self.p)
        x = self.partition_swap(x)
        x = x + self.stochastic_depth(self.attn_layer(x))
        x = x + self.stochastic_depth(self.mlp(self.norm(x)))
        x = self.departition_swap(x)
        x = self.departition_op(x, self.p, hp, wp)
        
        return x
    
class RelativePositionalMultiHeadAttn(nn.Module):
    def __init__(
        self,
        in_channels,
        head_dim,
        max_seq_len
    ):
        super().__init__()
        self.n_heads = in_channels // head_dim
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        
        self.qkv = nn.Linear(in_channels, in_channels*3)
        self.merge = nn.Linear(in_channels, in_channels)
        self.scale = self.head_dim**-0.5
        # do relative pos
        
    def forward(self, x):
        B, G, P ,D = x.shape
        H, DH = self.n_heads, self.head_dim
        
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        
        q = q.reshape(B, G, P, H, DH).permute(0,1,3,2,4)
        k = k.reshape(B, G, P, H, DH).permute(0,1,3,2,4)
        v = v.reshape(B, G, P, H, DH).permute(0,1,3,2,4)
        
        q = q * self.scale
        attn = torch.einsum("B G H I D, B G H J D -> B G H I J", q, k)
        
        # attn + relative_pos_bias
        
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("B G H I J, B G H J D -> B G H I D", attn, v)
        out = out.permute(0, 1, 3, 2, 5)
        out = out.reshape(B, G, P, D)
        
        out = self.merge(out)
        return out
    
class WindowPartition(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, x, p):
        B, C, H, W = x.shape
        num_windows = (H//p) * (W//p)
        x = x.reshape(B, C, H//p, p, W//p, p)
        x = x.premute(0,2,4,3,5,1)
        x = x.reshape(B, num_windows, p*p, C)
        return x        

class WindowDepartition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, p, hp, wp):
        B, HpWp, pp, C = x.shape
        
        x = x.reshape(B, hp, wp, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, hp*p, wp*p)
        return x
    
class SwapAxis(nn.Module):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def forward(self, x):
        x = torch.swapaxes(x, self.a, self.b)
        return x
