import math
from functools import partial

import torch
from torch import nn
import torchvision

from patch_merge import SimplePatchMerging
from s_swin_transformer_block import SimpleSwinTransformerBlock

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

# Shifted window Attention
# window_size=[4, 4]
# shift_size for depth = 2 : [0,0], [2,2] 
# num_heads = 3
# dim = 96

class SimpleSwinTransformer(nn.Module):
    """
        Implementation of swin transformer: https://arxiv.org/abs/2103.14030
    """
    def __init__(
        self,
        patch_size,
        emb_dim,
        depths, # List
        num_heads, # List
        window_size, # List
        mlp_ratio = 4.0,
        dropout = 0.0,
        attention_dropout = 0.0,
        stochastic_depth_prob = 0.1,
        num_classes = 1000,
        downsample_layer = SimplePatchMerging
    ):
        super().__init__()
        self.num_classes = num_classes
        
        block = SimpleSwinTransformerBlock
        norm_layer = partial(nn.LayerNorm, eps=1e-5)    

        layers = []
        layers.append(
            nn.Sequential(
                # patch the image with kernel size 4 and stride 4 => H/4, W/4
                nn.Conv2d(3, emb_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])),
                # (N, C, H, W) -> (N, H, W, C)
                torchvision.ops.misc.Permute([0,2,3,1]),
                norm_layer(emb_dim)     
            )
        )
       
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        
        for i_stage in range(len(depths)):
            stages = []
            dim = emb_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stages.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer
                    )
                )     
                stage_block_id += 1
            layers.append(nn.Sequential(*stages))          
            if i_stage < len(depths) - 1:
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)
        
        num_features = emb_dim * 2**(len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.permute = torchvision.ops.misc.Permute([0,3,1,2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(num_features, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x