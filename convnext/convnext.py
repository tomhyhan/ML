import torch
from torch import nn
from functools import partial
from torchvision.ops.misc import Conv2dNormActivation

from cnblock import CNBlock, CNBlockConfig
from layernorm2d import LayerNorm2d

# Changing stage compute ratio
#  3 3 9 3
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
        block_setting: list[CNBlockConfig],
        stochastic_depth_prob,
        layer_scale,
        num_classes,
        block,
        norm_layer,
    ):
        super().__init__()
        
        if block is None:
            block = CNBlock
        
        if norm_layer is None:
            # Substituting BN with LN
            norm_layer = partial(LayerNorm2d, eps=1e-6) 
        
        firstconv_output_channels = block_setting[0].input_channels
        
        layers = []
        # Changing stem to “Patchify”. 
        # 4x4 non-overlapping convolution
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True
            )
        )
        
        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            stage = []
            for _ in range(cnf.num_layers):
                sd_prob = stochastic_depth_prob * (stage_block_id / (total_stage_blocks - 1.0))
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id +=1
            layers.append(nn.Sequential(*stage))
            if cnf.output_channels is not None:
                # Separate downsampling layers. 
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.output_channels, kernel_size=2, stride=2)        
                    )
                )
                
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        lastblock = block_setting[-1]
        lastconv_output_channel = lastblock.output_channels if lastblock.output_channels is not None else lastblock.input_channels
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channel),
            nn.Flatten(1),
            nn.Linear(lastconv_output_channel, num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x