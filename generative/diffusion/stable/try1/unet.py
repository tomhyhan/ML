import torch
import torch.nn as nn
from blocks import DownBlock, UpBlock, MidBlock

class Unet(nn.Module):
    def __init__(
        self,
        im_channels,
        model_config
    ):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']
        
        self.class_cond = False
        self.text_cond = False
        self.image_cond = False
        self.text_emb_dim = None
        self.condition_config = model_config["condition_config"]
        
        if self.condition_config is not None:
            if 'class' in self.condition_config:
                self.class_cond = True
                self.num_classes = self.condition_config["class_condition_config"]["num_classes"]
            if 'text' in self.condition_config:
                self.text_cond = True
                self.text_emb_dim = self.condition_config["text_condition_config"]["text_emb_dim"]
            if 'image' in self.condition_config:
                self.image_cond = True
                self.im_cond_input_ch = self.condition_config["image_condition_config"]["image_condition_input_channel"]
                self.im_cond_output_ch = self.condition_config["image_condition_config"]["image_condition_output_channel"]
        
        if self.class_cond:
            self.class_emb = nn.Embedding(self.num_classes, self.t_emb_dim)
        if self.image_cond:
            self.cond_conv_in = nn.Conv2d(
                in_channels=self.im_cond_output_ch,
                out_channels=self.im_cond_input_ch,
                kernel_size=1,
                bias=False
            )
            self.conv_in_concat = nn.Conv2d(
                in_channels=im_channels+self.im_cond_input_ch,
                out_channels=self.down_channels[0],
                kernel_size=3,
                padding=1
            )
        else:
            self.conv_in = nn.Conv2d(
                in_channels=im_channels,
                out_channels=self.down_channels[0],
                kernel_size=3,
                padding=1
            )
        self.cond = self.text_cond or self.image_cond or self.class_cond
        
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        self.up_sample = list(reversed(self.down_sample))
        
        self.downs = nn.ModuleList()
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                DownBlock(
                    self.down_channels[i],
                    self.down_channels[i+1],
                    t_emb_dim=self.t_emb_dim,
                    num_layers=self.num_down_layers,
                    down_sample=self.down_sample[i],
                    num_heads=self.num_heads,
                    norm_channels=self.norm_channels,
                    attn=self.attns[i],
                    cross_attn=self.text_cond,
                    context_dim=self.text_emb_dim,
                )
            )
            
        self.mids = nn.ModuleList()
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                MidBlock(
                    self.mid_channels[i],
                    self.mid_channels[i+1],
                    t_emb_dim=self.t_emb_dim,
                    num_layers=self.num_mid_layers,
                    num_heads=self.num_heads,
                    cross_attn=self.text_cond,
                    context_dim=self.text_emb_dim,
                    norm_channels=self.norm_channels
                )
            )
            
        self.ups = nn.ModuleList()
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(
                UpBlock(
                    self.down_channels[i] * 2,
                    self.down_channels[i-1] if i != 0 else self.conv_out_channels,
                    t_emb_dim=self.t_emb_dim,
                    num_layers=self.num_down_layers,
                    down_sample=self.up_sample[i-1],
                    num_heads=self.num_heads,
                    attn=self.attns[i-1],
                    cross_attn=self.text_cond,
                    context_dim=self.text_emb_dim,
                    norm_channels=self.norm_channels,
                )
            )
                
        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t, cond_input):
        
        if self.image_cond:
            im_cond = cond_input["image"]
            im_cond = torch.nn.functional.interpolate(im_cond, size=x.shape[-2:])
            im_cond = self.cond_conv_in(im_cond)
            x = torch.concat([x, im_cond], dim=1)
            out = self.conv_in_concat(x)
        else:
            out = self.conv_in(x)
        
        t_emb = t
        t_emb = self.t_proj(t_emb)
        
        if self.class_cond:
            class_emb = cond_input["class"]
            class_emb = einsum(class_emb, self.class_emb.weight, 'B N, N D -> B D')
            t_emb += class_emb
            self.class_emb

        context_hidden_states = None
        if self.text_cond:
            context_hidden_states = cond_input["text"]

        down_outs = []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb, context_hidden_states)
            
        for mid in self.mids:
            out = mid(out, t_emb, context_hidden_states)
            
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb, context_hidden_states)
        
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        
        return out
