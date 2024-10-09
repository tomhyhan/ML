import torch
from torch import nn
from torchvision.ops.misc import MLP

from s_shifted_window_attn import ShiftedWindowAttention
from s_stochastic_depth import SimpleStochasticDepths

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
        
        self.norm2 = norm_layer(emb_dim)
        self.mlp = MLP(emb_dim, [int(emb_dim * mlp_ratio), emb_dim], activation_layer=nn.GELU, dropout=dropout)
        
        self.stochastic_depth = SimpleStochasticDepths(stochastic_depth_prob, "row")
        
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        
    def forward(self, x):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x    