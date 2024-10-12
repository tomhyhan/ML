import torch
from torch import nn
from shifted_window_attn1 import ShiftedWindowAttn
from torchvision.ops.misc import MLP
from torchvision.ops.stochastic_depth import StochasticDepth

class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim,
        window_size,
        shift_size,
        num_heads,
        mlp_ratio,
        stocastic_dropout_p,
        dropout,
        norm_layer
    ):
        self.norm_layer1 = norm_layer(emb_dim)
        self.norm_layer2 = norm_layer(emb_dim)
        
        self.attn = ShiftedWindowAttn(
            emb_dim,
            window_size,
            shift_size,
            num_heads,
            dropout
        )
        
        self.mlp = MLP(emb_dim, [int(emb_dim*mlp_ratio), emb_dim], activation_layer=nn.GELU, dropout=dropout)
        
        self.stocastic_dropout = StochasticDepth(p=stocastic_dropout_p, mode="row")
    
    def forward(self, x):
        x = self.stocastic_dropout(self.attn(self.norm_layer1(x)))
        x = self.stocastic_dropout(self.mlp(self.norm_layer2(x)))
        return x
