import torch
from torch import nn
from shifted_window_attn3 import ShiftedWindowAttn3
from torchvision.ops.misc import MLP
from torchvision.ops.stochastic_depth import StochasticDepth

class SwinBlock3(nn.Module):
    def __init__(
        self, 
        emb_dim,
        window_size,
        shift_size,
        num_head,
        mlp_ratio,
        dropout,
        attention_dropout,
        p_stochatic,
        norm_layer
    ) -> None:
        super().__init__()
        
        self.attn = ShiftedWindowAttn3(
            emb_dim,
            window_size,
            shift_size,
            num_head,
            dropout,
            attention_dropout
        )
        self.norm1 = norm_layer(emb_dim)
        
        self.mlp = MLP(emb_dim, [emb_dim*mlp_ratio, emb_dim], activation_layer=nn.GELU, dropout=dropout)
        self.norm2 = norm_layer(emb_dim)
        
        self.stochastic_depth = StochasticDepth(p=p_stochatic, mode="row") 
        
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        
    def forward(self, x):
        x = self.stochastic_depth(self.attn(self.norm1(x)))
        x = self.stochastic_depth(self.mlp(self.norm2(x)))
        return x