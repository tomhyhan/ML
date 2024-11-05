import torch
from torch import nn

# block_setting = [
#     CNBlockConfig(96, 192, 3),
#     CNBlockConfig(192, 384, 3),
#     CNBlockConfig(384, 768, 9),
#     CNBlockConfig(768, None, 3),
# ]
# stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)

class ConvNeXt(nn.Module):
    def __init__(
        self,
        block_setting,
        stochastic_depth_prob,
        layer_scale,
        num_classes,
        block,
        norm_layer,
    ):
        super().__init__()
        
        if block is None:
            block = CNBlock
        
        