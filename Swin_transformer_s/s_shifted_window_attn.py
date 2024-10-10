import torch
from torch import nn
import torch.nn.functional as F

class ShiftedWindowAttention(nn.Module):
    
    def __init__(
        self, 
        dim,
        window_size,
        shift_size,
        num_heads,
        qkv_bias = True,
        proj_bias = True,
        attention_dropout = 0.0,
        dropout = 0.0
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        
        self.define_relative_position_bias_table()
        self.define_relative_position_index()
        
    def define_relative_position_bias_table(self):
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        
    def get_relative_position_bias(self):
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size    
        )
        
    def forward(self, x):
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training
        ) 
        
def _get_relative_position_bias(
    relative_position_bias_table,
    relative_position_index,
    window_size
):
    N = window_size[0] * window_size[1]
    # 
    relative_position_bias = relative_position_bias_table[relative_position_index]
    relative_position_bias = relative_position_bias.view(N,N,-1)
    relative_position_bias = relative_position_bias.permute(2,0,1).contiguous().unsqueeze(0)
    return relative_position_bias
    
    
def shifted_window_attention(
    input,
    qkv_weight,
    proj_weight,
    relative_position_bias,
    window_size,
    num_heads,
    shift_size,
    attention_dropout=0.0,
    dropout=0.0,
    qkv_bias=None,
    proj_bias=None,
    training=True
):
    B, H, W, C = input.shape
    # 7 - 16 % 7
    pad_r = (window_size[1] - H % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - W % window_size[0]) % window_size[0]
    x = F.pad(input, (0,0,0,pad_r,0,pad_b))
    _, pad_H, pad_W, _ = x.shape
    
    shift_size = shift_size.copy()
    
    if window_size[0] > shift_size[0]:
        shift_size[0] = 0
    if window_size[1] > shift_size[1]:
        shift_size[1] = 0

    # cyclic shift: rolling
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1,2))
    
    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    # N, seq_len, emb
    x = x.permute(0,1,3,2,4,5).reshape(B * num_windows, window_size[0] * window_size[1], C)
    
    # multi-head attention
    qkv = F.linear(x, qkv_weight, qkv_bias)
    
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    # N, num_heads, seq_len, C 
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    q = q * (C // num_heads) ** -0.5
    # N, num_heads, seq_len, seq_len 
    attn = q.matmul(k.transpose(-2,-1))
    
    attn = attn + relative_position_bias
    
    if sum(shift_size) > 0:
        # masked attention mask is required to avoid unrelated adjacent patches to share data
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slice = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slice = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))

        cnt = 0
        for h in h_slice:
            for w in w_slice:
                attn_mask[h[0] : h[1], w[0]: w[1]]
                cnt += 1
                
        attn_mask = attn_mask.view(pad_W // window_size[0], window_size[0], pad_H // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0,2,1,3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        # shape - num_windows, seq_len, seq_len
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0,0))
        
        # x - N, seq_len, emb
        # shape - N, num_windows, num_heads, seq_len, seq_len
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))
        
    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)
    
    x = attn.matmul(v).transpose(1,2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)
    
    # reverse windows
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0,1,3,2,4,5).reshape(B, pad_W, pad_H, C)
    
    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        
    x = x[:, :H, :W, :].contiguous()
    
    return x