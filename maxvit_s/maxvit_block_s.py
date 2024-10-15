from torch import nn
from utils import conv_out_size
from maxvit_layer_s import SimpleMaxVITLayer

class SimpleMaxvitBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        sqeeze_ratio,
        expansion_ratio,
        norm_layer,
        activation_layer,
        head_dim,
        mlp_ratio,
        mlp_dropout,
        attention_dropout,
        partition_size,
        input_grid_size,
        n_layers,
        p_stochastic
    ):
        super().__init__()
        
        if not len(p_stochastic) == n_layers:
            raise ValueError(f"p_stochastic must have length n_layers={n_layers}, got p_stochastic={p_stochastic}.")
        
        self.layers = nn.ModuleList()
        self.grid_size = conv_out_size(input_grid_size)
        
        for idx, p in enumerate(p_stochastic):
            stride = 2 if idx == 0 else 1
            self.layers += [
                SimpleMaxVITLayer(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    sqeeze_ratio=sqeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    grid_size=self.grid_size,
                    p_stochastic_dropout=p
                )
            ]
    
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x