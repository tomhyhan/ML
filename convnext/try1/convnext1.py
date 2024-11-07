from torch import nn
from functools import partial
from torchvision.ops.misc import Conv2dNormActivation

from layernorm2d1 import LayerNorm2d1
from cnblock1 import CNBlock1, CNBlockSetting1

class ConvNext1(nn.Module):
    def __init__(
        self,
        block_settings: list[CNBlockSetting1],
        block,
        norm_layer,
        stochatic_depth_prob,
        layer_scale,
        num_classes
    ):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d1, eps=1e-6)
            
        if block is None:
            block = CNBlock1
            
        firstconv_output_channels = block_settings[0].input_channels
        
        layers = []
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True
            )            
        )    
            
        total_blocks = sum(cns.num_layers for cns in block_settings)
        stage_id = 0
        
        for cns in block_settings:
            stage = []
            for _ in range(cns.num_layers):
                sd_prob = stochatic_depth_prob * (stage_id / (total_blocks - 1))
                stage.append(
                    block(
                        cns.input_channels,
                        layer_scale,
                        sd_prob 
                    )                    
                )
                stage_id += 1
            layers.append(nn.Sequential(*stage))
            if cns.output_channels is not None:
                layers.append(
                    nn.Sequential(
                        norm_layer(cns.input_channels),
                        nn.Conv2d(
                            cns.input_channels,
                            cns.output_channels,
                            kernel_size=2,
                            stride=2
                        )
                    )
                )
                
        last_block = block_settings[-1]
        last_conv_dim = last_block.output_channels if last_block.output_channels is not None else last_block.input_channels
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            norm_layer(last_conv_dim),
            nn.Flatten(1),
            nn.Linear(last_conv_dim, num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
