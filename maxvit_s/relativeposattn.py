import math
import torch
from torch import nn
import torch.nn.functional as F

class RelativePositionalMultiHeadAttn(nn.Module):
    def __init__(
        self,
        feat_dim,
        head_dim,
        max_seq_len
    ):
        super().__init__()
        
        if feat_dim % head_dim != 0:
            raise ValueError(f"feat_dim: {feat_dim} must be divisible by head dim")

        self.n_heads = feat_dim // head_dim
        self.head_dim = head_dim
        self.partition = int(math.sqrt(max_seq_len))
        self.max_seq_len = max_seq_len
        
        self.to_qkv = nn.Linear(feat_dim, self.n_heads * self.head_dim * 3)
        self.scale_factor = head_dim**-0.5
        
        self.merge = nn.Linear(self.head_dim * self.n_heads, feat_dim)
        self.relative_pos_table = nn.Parameter(
            torch.zeros((2*self.partition-1) * (2*self.partition-1), self.n_heads)
        )
        self.relative_pos_index = self._get_relative_pos_index(self.partition, self.partition)

    def _get_relative_pos_index(self, h, w):
        H = torch.arange(h)
        W = torch.arange(w)
        coords = torch.stack(torch.meshgrid(H,W, indexing="ij"))
        coords_flatten = coords.flatten(1)
        coords_flatten = coords_flatten[:,:,None] - coords_flatten[:,None,:]
        coords_perm = coords_flatten.permute(1,2,0)
        coords_perm[:,:,0] += h -1
        coords_perm[:,:,1] += w -1
        coords_perm[:,:,0] *= 2*h -1
        index = coords_perm.sum(-1).flatten()
        return index
    
    def get_relatove_pos_bias(self):
        pos_bias = self.relative_pos_table[self.relative_pos_index]
        pos_bias = pos_bias.reshape(self.max_seq_len, self.max_seq_len, -1)
        pos_bias = pos_bias.permute(2,0,1)
        return pos_bias.unsqueeze(0)
    
    def forward(self, x):
        B, G, P, D = x.shape
        H, DH = self.n_heads, self.head_dim
        
        qkv = self.to_qkv(x)
        q,k,v = torch.chunk(qkv, 3, dim=-1)
        
        q = q.reshape(B, G, P, H, DH).permute(0,1,3,2,4)        
        k = k.reshape(B, G, P, H, DH).permute(0,1,3,2,4)        
        v = v.reshape(B, G, P, H, DH).permute(0,1,3,2,4)
        
        q = q * self.scale_factor
        attn = torch.einsum("B G H I D, B G H J D -> B G H I J", q, k)
        pos_bias = self.get_relatove_pos_bias()
        
        attn = F.softmax(attn + pos_bias, dim=-1)

        out = torch.einsum("B G H I J, B G H J D -> B G H I D", attn, v)
        out = out.permute(0,1,3,2,4).reshape(B,G,P,D)
        
        out = self.merge(out)
        return out