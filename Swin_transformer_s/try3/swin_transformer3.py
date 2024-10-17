import torch
from torch import nn
from functools import partial
from torchvision.ops.misc import Permute

from swin_block3 import SwinBlock3
from patch_merge3 import PatchMerge3

class SwinTransformer3(nn.Module):
    def __init__(
        self,
        emb_dim,
        init_patch_size,
        depths,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.1,
        stochatic_depth_p=0.2,
        num_classes=10
    ):
        super().__init__()
        
        in_channels = 3
        
        norm_layer = partial(nn.LayerNorm, 1e-5) 
        block = SwinBlock3
        
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    emb_dim,
                    init_patch_size,
                    init_patch_size
                ),
                Permute([0,2,3,1])
            ),
        )
        
        total_depths = sum(depths)
        stage_id = 0
        
        for layer_i in range(len(depths)):
            stages = []
            dim = emb_dim * 2**layer_i
            for block_i in range(depths[layer_i]):
                p_stochatic = stochatic_depth_p * (stage_id / (total_depths-1))
                stages.append(
                    block(
                        dim,
                        window_size=window_size,
                        shift_size=[0 if block_i % 2 == 0 else w // 2 for w in window_size],
                        num_head=num_heads[block_i],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        p_stochatic=p_stochatic,
                        norm_layer=norm_layer
                    )
                )
                stage_id += 1
            layers.append(nn.Sequential(*stages))
            if layer_i < len(depths) - 1:
                layers.append(PatchMerge3(dim, norm_layer))
            
        self.features = nn.Sequential(*layers)
        
        dim = emb_dim * 2**(len(depths)-1)
        self.norm = norm_layer(dim)
        
        self.permute = Permute([0,3,1,2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x