import math
import torch
from torch import nn
import torch.nn.functional as F

class RelativePositionalMultiHeadAttn2(nn.Module):
    def __init__(
        self,
        in_channels,
        head_dim,
        max_seq_len
    ):
        self.n_heads = in_channels // head_dim
        self.partition_size = int(math.sqrt(max_seq_len))
        self.attn_scale = head_dim**-0.5
        self.head_dim = head_dim
        
        self.qkv = nn.Linear(in_channels, in_channels*3)
        self.merge = nn.Linear(in_channels, in_channels)
        
        self._get_relative_pos_table()
        self._get_relative_pos_index()
        
        relative_pos_bias = self.relative_pos_table[self.relative_position_index].reshape(max_seq_len, max_seq_len, -1)
        relative_pos_bias = relative_pos_bias.permute(2,0,1)
        self.relative_pos_bias = relative_pos_bias.unsqueeze(0)
        
    def _get_relative_pos_table(self):
        self.relative_pos_table= nn.Parameter(
            torch.zeros((2*self.partition_size-1)*(2*self.partition_size-1), self.n_heads) 
        )
        
    def _get_relative_pos_index(self):
        H = torch.arange(self.partition_size)
        W = torch.arange(self.partition_size)
        coords = torch.stack(torch.meshgrid(H, W, indexing="ij"))
        coords_flatten = coords.flatten(1)
        coords_flatten = coords_flatten[:,:,None] - coords_flatten[:,None,:]
        coords_perm = coords_flatten.permute(1,2,0)
        coords_perm[:,:,0] += self.partition_size - 1
        coords_perm[:,:,1] += self.partition_size - 1
        coords_perm[:,:,0] *= 2*self.partition_size - 1

        relative_position_index = coords_perm.sum(-1).flatten()
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, x):
        B, G, P, D = x
        
        H, DH= self.n_heads, self.head_dim
        
        qkv = self.qkv(x)
        q,k,v = torch.chunk(qkv, chunks=3, dim=-1)
        
        q = q.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        
        q = q * self.attn_scale
        attn = torch.einsum("B G H I D, B G H J D -> B G H I J", q, k)
        
        # difference btw registering
        attn = attn + self.relative_pos_bias
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum("B G H I J, B G H J D -> B G H I D", attn, v)
           
        out = out.permute(0,1,3,2,4).reshape(B, G, P, D)     
        out = self.merge(out)
        return out
        