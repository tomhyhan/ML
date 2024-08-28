import torch
from torch import nn
from s_shifted_window_attn import ShiftedWindowAttention

class SimpleSwinTransformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio = 4.0,
        dropout = 0.0,
        attention_dropout = 0.0,
        stochastic_depth_prob = 0.0,
        norm_layer = nn.LayerNorm,
        attn_layer = ShiftedWindowAttention
    ):
        super().__init__()
        self.norm1 = norm_layer(emb_dim)
        self.attn = attn_layer(
            emb_dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout
        )
        self.stochastic_depth = StochasticDepth
        
    def forward(self):
        pass    
    