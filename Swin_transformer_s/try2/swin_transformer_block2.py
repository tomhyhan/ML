from torch import nn
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.ops.misc import MLP
from shifted_window_attn2 import ShiftedWindowAttn2

class SwinTranformerBlock2(nn.Module):
    def __init__(
        self,
        emb_dim,
        window_size,
        shift_size,
        num_heads,
        stochastic_p,
        dropout,
        mlp_ratio,
        norm_layer,
    ):
        super().__init__()

        self.msa = ShiftedWindowAttn2(
            emb_dim,
            window_size,
            shift_size,
            num_heads,
            dropout,
        )
        self.norm_layer1 = norm_layer(emb_dim)
        
        self.mlp = MLP(emb_dim, [emb_dim, emb_dim*mlp_ratio], activation_layer=nn.GELU, dropout=dropout)
        self.norm_layer2 = norm_layer(emb_dim)
        self.stochastic = StochasticDepth(p=stochastic_p, mode="row")

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        
    def forward(self, x):
        x = self.stochastic(self.norm_layer1(self.msa(x)))
        x = self.stochastic(self.norm_layer1(self.mlp(x)))
        return x
