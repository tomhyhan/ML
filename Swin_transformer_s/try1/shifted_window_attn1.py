import torch 
from torch import nn
import  torch.nn.functional as F

class ShiftedWindowAttn(nn.Module):
    
    def __init__(
        self,
        emb_dim,
        window_size,
        shift_size,
        num_heads,
        dropout,
        qkv_bias=True,
        proj_bias=True
    ):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.qkv = nn.Linear(emb_dim, emb_dim*3, bias=qkv_bias)
        self.proj = nn.Linear(emb_dim, emb_dim, bias=proj_bias)
        
        relative_position_table = self.get_relative_position_table()
        relative_position_index = self.get_relative_position_index()
        
        num_windows = window_size[0] * window_size[1]
        self.relative_position = relative_position_table[relative_position_index] \
            .view(num_windows, num_windows, -1) \
            .permute(2,0,1) \
            .unsqueeze(0) # for batch
        
    def get_relative_position_table(self):
        relative_position_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[0] - 1), self.num_heads)
        )
        nn.init.trunc_normal_(relative_position_table, std=0.02)
        return relative_position_table
    
    def get_relative_position_index(self):
        H = torch.arange(self.window_size[0])
        W = torch.arange(self.window_size[1])
        coords = torch.meshgrid(H, W, indexing="ij")
        coords_stack = torch.stack(coords)
        coords_flatten = torch.flatten(coords_stack, dims=1)
        coords_flatten = coords_flatten[:,:,None] - coords_flatten[:,None,:]
        coords_perm = coords_perm.permute(1,2,0)
        
        coords_perm[:,:,0] += self.window_size[0] - 1
        coords_perm[:,:,1] += self.window_size[0] - 1
        coords_perm[:,:,0] *= 2*self.window_size[0] - 1

        coords_index = coords_perm.sum(-1).flatten()
        return coords_index
    
    def forward(self, x):
        return shifted_window(
            x,
            self.qkv.weight,
            self.proj.weight,
            self.emb_dim,
            self.window_size,
            self.shift_size,
            self.num_heads,
            self.dropout,
            self.relative_position,
            self.qkv.bias,
            self.proj.bias,
            self.training            
        )

def shifted_window(
    x : torch.Tensor,
    qkv_weight,
    proj_weight,
    emb_dim,
    window_size,
    shift_size,
    num_heads,
    dropout,
    relative_position,
    qkv_bias,
    proj_bias,
    training
):
    _, H, W, _ = x.shape
    pad_r = (window_size[0] - (H % window_size[0])) % window_size[0] # to make pad_r 0 when (window_size[0] - (H % window_size[0])) is same as window_size[0]
    pad_b = (window_size[1] - (W % window_size[1])) % window_size[1]
    # add padding to right and bottom
    x = F.pad(x, (0,0,0,pad_r,0,pad_b,0,0))
    N, pad_H, pad_W, C = x
    
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 1
    
    # shifted window using roll
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        

    # tokenize the patch
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(N, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0,1,3,2,4,5)
    # batch_size, seq_len, emb_dim
    x = x.reshape(N*num_windows, window_size[0] * window_size[1], C)
    
    # multi_head attn block
    qkv = nn.Linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3,num_heads, C // num_heads)
    # 3, batch_size, num_heads, seq_len, emb_dim
    qkv = qkv.premute(2, 0, 3, 1, 4)
    # batch_size, num_heads, seq_len, emb_dim
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    q = q * (C // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2,-1))
    
    attn = attn + relative_position
    
    # attntion mask
    if sum(shift_size) > 0:
        attn_mask = torch.zeros(pad_H, pad_W)
        h_slice = [(0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None)]
        w_slice = [(0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None)]

        cnt = 0
        for h in h_slice:
            for w in w_slice:
                attn_mask[h[0]:h[1], w[0]:w[1]] = cnt
                cnt += 1
                
        attn_mask = attn_mask.reshape(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.premute(0,2,1,3)
        attn_mask = attn_mask.reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask[:,:,None] - attn_mask[:,None,:]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask[None, : , None, :, :]
        attn = attn.reshape(-1, num_heads, x.size(1), x.size(1))
    
    attn = F.softmax(attn, dim=-1)

    # N*num_windows, num_heads, seq_len, C // emb_dim
    # head was combined to C
    # N*num_windows, seq_len, emb_dim
    x = attn.matmul(v).transpose(1,2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)
    
    x = x.reshape(N, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C).premute(0,1,3,2,4,5)
    x = x.reshape(N, pad_H, pad_W, C)
    
    if sum(shift_size) > 0:
        x = x.roll(shifts=(shift_size[0], shift_size[1]), dim=(1,2))
        
    return x[:, :H, :W, :].contiguous()
