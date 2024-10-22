import numpy as np
from torch import nn
from functools import partial
from torchvision.ops.misc import Conv2dNormActivation

from maxvit_block_s import SimpleMaxvitBlock
from utils import make_block_input_shapes, conv_out_size

# stem_channels=64,
# block_channels=[64, 128, 256, 512],
# block_layers=[2, 2, 5, 2],
# head_dim=32,
# stochastic_depth_prob=0.2,
# partition_size=7,
# weights=weights,
# progress=progress,

class SimpleMaxVIT(nn.Module):
    def __init__(
        self,
        input_size,
        stem_channels,
        partition_size,
        block_channels,
        block_layers,
        head_dim,
        stochastic_depth_prob,
        norm_layer = None,
        activation_layer = nn.GELU,
        sqeeze_ratio=0.25,
        expansion_ratio=4,
        mlp_ratio=4,
        mlp_dropout=0.0,
        attention_dropout=0.0,
        num_classes=10
    ):
        super().__init__()
        
        input_channels = 3

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        
        block_input_sizes = make_block_input_shapes(input_size, len(block_channels))
        for idx, block_input_size in enumerate(block_input_sizes):
            if block_input_size[0] % partition_size != 0 or block_input_size[1] % partition_size != 0:
                raise ValueError(
                    f"Input size {block_input_size} of block {idx} is not divisible by partition size {partition_size}. "
                    f"Consider changing the partition size or the input size.\n"
                    f"Current configuration yields the following block input sizes: {block_input_sizes}."
                )
    
        # first stem layer
        self.stem = nn.Sequential(
            Conv2dNormActivation(
                input_channels,
                stem_channels,
                3,
                2,
                norm_layer=norm_layer,
                activation_layer=nn.GELU,
                bias=False,
                inplace=None
            ),
            Conv2dNormActivation(
                stem_channels,
                stem_channels,
                3,
                1,
                norm_layer=None,
                activation_layer=None,
                bias=True
            )
        )
        
        # account for stem stride
        input_size = conv_out_size(input_size, 3, 2, 1)
        self.partition_size = partition_size
        
        self.blocks = nn.ModuleList()
        in_channels = [stem_channels] + block_channels[:-1]
        out_channels = block_channels
        
        p_stochastic = np.linspace(0, stochastic_depth_prob, sum(block_layers)).tolist()
        
        p_idx = 0
        
        for in_channel, out_channel, num_layers in zip(in_channels, out_channels, block_layers):
            self.blocks.append(
                SimpleMaxvitBlock(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    sqeeze_ratio=sqeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    input_grid_size=input_size,
                    n_layers=num_layers,
                    p_stochastic=p_stochastic[p_idx: p_idx+num_layers]
                )
            )
            input_size = self.blocks[-1].grid_size
            p_idx += num_layers

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(block_channels[-1]),
            nn.Linear(block_channels[-1], block_channels[-1]),
            nn.Tanh(),
            nn.Linear(block_channels[-1], num_classes, bias=False)           
        )
        
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return x