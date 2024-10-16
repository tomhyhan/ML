import torch
from torch import nn
from torchvision.ops.misc import Permute
from functools import partial

from patch_merge2 import PatchMerge2
from try2.swin_transformer_block2 import SwinTranformerBlock2

class SwinTranformer2(nn.Module):
    def __init__(
        self,
        emb_dim,
        init_patch_size,
        depths,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        stochastic_depth_p=0.2,
        dropout=0.1,
        num_classes=10
    ):
        super().__init__()
        in_channels = 3
        block = SwinTranformerBlock2
        norm_layer = partial(nn.LayerNorm, eps=1e-5)
        
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    emb_dim,
                    init_patch_size,
                    init_patch_size,
                ),
                Permute([0,2,3,1]),
                norm_layer(emb_dim)
            )
        )
        
        n_depths = sum(depths)
        stage_id = 0
        for i in range(len(depths)):
            stages = []
            dim = emb_dim * 2 ** i
            for stage in depths[i]:
                p_stochatic = stochastic_depth_p * (stage_id / (n_depths - 1)) 
                stages.append(
                    block(
                        dim,
                        window_size,
                        shift_size=[0 if stage % 2 == 0 else w // 2 for w in window_size],
                        num_heads=num_heads[stage],
                        p_stochatic=p_stochatic,
                        dropout=dropout,
                        mlp_ratio=mlp_ratio,
                        norm_layer=norm_layer
                    )
                )
                stage_id += 1
            layers.append(nn.Sequential(*stages))
            if i < len(depths) - 1:
                stages.append(PatchMerge2(dim, norm_layer))
        self.features = nn.Sequential(*layers) 
        
        emb_dim = emb_dim * 2 ** (len(depths) - 1)
        self.norm_layer = norm_layer(emb_dim)
        self.permute = Permute([0,3,1,2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(emb_dim, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.norm_layer(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x