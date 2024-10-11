from torch import nn
from torchvision.ops.misc import Permute
from swin_transformer_block1 import SwinTransformerBlock
from patch_merging import PatchMerge

class SwinTransformer1(nn.Module):
    # patch_size = [4,4]
    # emb_dim = 96
    # window_size = [7,7]
    # depths = [2,2,6,2]
    # num_heads = [2,4,8,16]
    # stocastic_dropout = 0.1
    # dropout = 0.1
    # mlp_ratio = 4
    def __init__(
        self,
        patch_size,
        emb_dim,
        window_size,
        depths, # list
        num_heads, # list
        mlp_ratio,
        stocastic_dropout,
        dropout=0.1,
    ):
        super().__init__()
        
        block = SwinTransformerBlock
        norm_layer = nn.LayerNorm
        
        layers = []
        layers.append(
            nn.Sequential([
                nn.Conv2d(3, emb_dim, patch_size[0], patch_size[1]),
                Permute([0,2,3,1])
            ])
        )
        
        total_depth = sum(depths)
        depth_id = 0
        
        for i in range(len(depths)):
            sw_blocks = []
            emb_dim *= 2 ** i
            for depth in depths[i]:
                s_dropout = stocastic_dropout * (depth_id / (total_depth - 1))
                sw_blocks.append(
                    # fill this in
                    block()
                )
            depth_id += 1
            layers.append(nn.Sequential(*sw_blocks))
            if i < len(depth) - 1:
                layers.append(PatchMerge(emb_dim, norm_layer))
        # processing Swin Transformer blocks
        self.features = nn.Sequential(*layers)



    def forward(self):
        pass