from torch import nn

class SimpleMaxVITLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        sqeeze_ratio,
        expansion_ratio,
        stride,
        norm_layer,
        activation_layer,
        head_dim,
        mlp_ratio,
        mlp_dropout,
        attention_dropout,
        partition_size,
        grid_size,
        p_stochastic_dropout
    ):
        super().__init__()
        pass
    
    def forward(self):
        pass
