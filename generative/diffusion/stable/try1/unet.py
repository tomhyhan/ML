import torch
import torch.nn as nn

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
                