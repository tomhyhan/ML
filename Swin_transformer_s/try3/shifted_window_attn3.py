import torch
from torch import nn
import torch.nn.functional as F
class ShiftedWindowAttn3(nn.Module):
    def __init__(
        self,
        emb_dim,
        window_size,
        shift_size,
        num_head,
        dropout,
        attention_dropout,
        qkv_bias=True,
        proj_bias=True 
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_head = num_head
        self.dropout = dropout
        self.attention_dropout = attention_dropout        
    
        self.qkv = nn.Linear(emb_dim, emb_dim*3, bias=qkv_bias)
        self.proj = nn.Linear(emb_dim, emb_dim, bias=proj_bias)

        self.relative_pos_table = self._get_relative_pos_table()
        self.relative_pos_index = self._get_relative_pos_index()

        nn.init.trunc_normal_(self.relative_pos_table, std=0.02)

        num_windows = self.window_size[0] * self.window_size[1]
        relative_pos_bias = self.relative_pos_table[self.relative_pos_index]
        relative_pos_bias = relative_pos_bias.reshape(num_windows, num_windows, num_head)
        relative_pos_bias = relative_pos_bias.permute(2,1,0).unsqueeze(0)
        self.relative_pos_bias = relative_pos_bias
        
    def _get_relative_pos_table(self):
        return nn.Parameter(
            torch.zeros((2*self.window_size[0]-1)*2*self.window_size[1]-1, self.num_head)
        )
    
    def _get_relative_pos_index(self):
        H = torch.arange(self.window_size[0])
        W = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(H,W,indexing='ij'))
        coords_flatten = coords.flatten(1)
        coords_flatten = coords_flatten[:,:,None] - coords_flatten[:,None,:]
        coords_perm = coords_flatten.permute(1,2,0)
        coords_perm[:,:,0] += self.window_size[0] - 1
        coords_perm[:,:,1] += self.window_size[0] - 1
        coords_perm[:,:,0] *= 2*self.window_size[0] - 1
        index = coords_perm.sum(-1).flatten()
        return index
    
    def forward(self, x):
        return shifted_window_attn(
            x,
            self.emb_dim,
            self.window_size,
            self.shift_size,
            self.num_head,
            self.dropout,
            self.attention_dropout,
            self.relative_pos_bias,
            self.qkv.weight,
            self.proj.weight,
            self.qkv.bias,
            self.proj.bias
        )

def shifted_window_attn(
    x,
    emb_dim,
    window_size,
    shift_size,
    num_head,
    dropout,
    attention_dropout,
    relative_pos_bias,
    qkv_weight,
    proj_weight,
    qkv_bias,
    proj_bias
):
    N, H, W, C = x.shape
    pad_r = (window_size[0] - H % window_size[0] ) % window_size[0]
    pad_b = (window_size[1] - H % window_size[1] ) % window_size[1]
    x = F.pad(x, (0,0,0,pad_r,0,pad_b,0,0))
    _, pad_H, pad_W, _ = x.shape
    
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1,2))
    
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.reshape(N, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0,1,3,2,4,5)
    # N, seq_len, emb_dim
    x = x.reshape(N*num_windows, window_size[0] * window_size[1], C)
    
    # N, seq_len, 3*emb_dim
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_head, C // num_head)
    # 3, N, num_head, seq_len, emb_dim
    qkv = qkv.permute(2,0,3,1,4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    q = q * (C // num_head)**-0.5
    attn = q.matmul(k.transpose(-1,-2))
    attn = attn + relative_pos_bias
    
    if sum(shift_size) > 0:
        attn_mask = torch.zeros(pad_H, pad_W)
        h_slice = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slice = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        
        cnt = 0
        for h in h_slice:
            for w in w_slice:
                attn_mask[h[0]:h[1], w[0]:w[1]] = cnt
                cnt += 1
        
        attn_mask = attn_mask.reshape(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0,2,1,3).reshape(num_windows, window_size[0] * window_size[1])
        # num_windows, seq_len, seq_len
        attn_mask = attn_mask[:,:,None] - attn_mask[:,None,:]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        attn = attn.reshape(N, num_windows, num_head, x.size(1), x.size(1))
        attn = attn + attn_mask[None, :, None, :, :]
        attn = attn.reshape(-1, num_head, x.size(1), x.size(1))
        
    # N*num_window, seq_len, num_head, emb_dim // num_head
    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)
    
    attn = attn.matmul(v).transpose(1,2).reshape(x.size(0), x.size(1), -1)
    
    x = F.linear(attn, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout)
    
    x = x.reshape(N, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.premute(0,1,3,2,4,5)
    x = x.reshape(N, pad_H, pad_W, C)
    
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1,2))
        
    return x[:, :H, :W, :].contiguous()
    
    