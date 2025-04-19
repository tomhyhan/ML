import torch.nn as nn
from model.blocks import MidBlock, DownBlock, UpBlock

class VAE(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config["down_channels"]
        self.mid_channels = model_config["mid_channels"]
        self.down_sample = model_config["down_sample"]
        self.num_down_layers = model_config["num_down_layers"]
        self.num_mid_layers = model_config["num_mid_layers"]
        self.num_up_layers = model_config["num_up_layers"]
        
        self.attns = model_config["attn_down"]
        
        self.z_channels = model_config["z_channels"]
        self.norm_channels = model_config["norm_channels"]
        self.num_heads = model_config["num_heads"]

        # define up samples from down samples
        self.up_sample = list(reversed(self.down_channels))

        # Encoder
        # define entry point conv in for image to down channels
        self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0], 3, 1, 1)
        
        # define down layer and mid layer
        self.down_layers = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.down_layers.append(
                DownBlock(self.down_channels[i], self.down_channels[i+1], t_emb_dim=None, down_sample=self.down_sample[i], num_heads=self.num_heads, num_layers=self.num_down_layers, attn=self.attns[i], norm_channels=self.norm_channels)
            )

        self.mid_layers = nn.ModuleList([])
        for i in range(len(self.mid_channels)):
            self.mid_layers.append(
                MidBlock(self.down_channels[-1], self.mid_channels[i], t_emb_dim=None, num_heads=self.num_heads, norm_channels=self.norm_channels)
            )
        
        # normalize output from mid block
        self.encover_norm = nn.GroupNorm(self.norm_channels, self.mid_channels[-1])        
        # conv out from mid channel to z channel * 2 (for mean and variance)
        
        # define pre quant conv
        
        # Decoder
        # define post quant conv
        
        # define entry point decoder conv in for z to mid channel
        
        # define mid layer
        
        # define decoder layer
        
        # normalize decoder output
        
        # conv out from decoder channel to im_channels