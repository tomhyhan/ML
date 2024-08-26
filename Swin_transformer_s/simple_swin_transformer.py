import torch
from torch import nn

# Model params
# patch_size=[4, 4],
# embed_dim=96,
# depths=[2, 2, 6, 2],
# num_heads=[3, 6, 12, 24],
# window_size=[7, 7],
# stochastic_depth_prob=0.2,
# weights=weights,
# progress=progress,
# **kwargs,

# Shifted window Attention
# window_size=[7, 7]
# shift_size for depth = 2 : [0,0], [3,3] 
# num_heads = 3

class SimpleSwinTransformer(nn.Module):
    """
        Implementation of swin transformer: https://arxiv.org/abs/2103.14030
        
    """
    def __init__(self):
        
        super().__init__()
        
        
    def forward(self):
        pass