from torch import nn

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels,
        t_emb_dim,
        num_layers,
        attn,
        num_heads,
        down_sample,
        cross_attn,
        context_dim,
        norm_channels,
    ):
        super().__init__()

        self.resnet_conv_1 = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1)
            ) for i in range(num_layers)
        )
        
        if t_emb_dim is not None:
            self.t_emb_layer = nn.ModuleList(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                ) for _ in range(num_layers)
            )
        
        self.resnet_conv_2 = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ) for i in range(num_layers)
        )
        
        if attn:
            self.attn_norms = nn.ModuleList(
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)
            )
            self.attention = nn.ModuleList(
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            )
            
        if cross_attn:
            self.cross_attn_norms = nn.ModuleList(
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)
            )
            self.cross_attention = nn.ModuleList(
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            )
            self.context_proj = nn.ModuleList(
                nn.Linear(context_dim, out_channels)
                for _ in range(num_layers)
            ) 
        
class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels,
        t_emb_dim,
        num_layers,
        attn,
        num_heads,
        down_sample,
        cross_attn,
        context_dim,
        norm_channels,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.t_emb_layer = t_emb_dim
        self.attn = attn
        self.cross_attn = cross_attn
        self.context_dim = context_dim

        self.resnet_conv_1 = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1)
            ) for i in range(num_layers)
        )
        
        if t_emb_dim is not None:
            self.t_emb_layer = nn.ModuleList(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                ) for _ in range(num_layers)
            )
        
        self.resnet_conv_2 = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ) for i in range(num_layers)
        )
        
        if attn:
            self.attn_norms = nn.ModuleList(
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)
            )
            self.attention = nn.ModuleList(
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            )
            
        if cross_attn:
            self.cross_attn_norms = nn.ModuleList(
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)
            )
            self.cross_attention = nn.ModuleList(
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            )
            self.context_proj = nn.ModuleList(
                nn.Linear(context_dim, out_channels)
                for _ in range(num_layers)
            )
        
        self.residual_add_layer = nn.ModuleList(
                nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            )
        
        self.down_sample_layer = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if down_sample else nn.Identity()
        
    def forward(self, x, t_emb=None, context=None):
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_1[i](out)
            if t_emb is not None:
                out = out + self.t_emb_layer[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_2[i](out)
            out = out + self.residual_add_layer[i](resnet_input)
            
            if self.attn:
                batch_size, channel, h, w = out.shape()
                attn_in = out.reshape(batch_size, channel, h*w)
                attn_in = self.attn_norms[i](out)
                attn_in = out.transpose(1,2)
                attn_out = self.attention[i](attn_in, attn_in, attn_in)
                attn_out = attn_out.transpose(1,2).reshape(batch_size, channel, h, w)
                out = out + attn_out
                
            if self.cross_attn:
                batch_size, channel, h, w = out.shape()
                attn_in = out.reshape(batch_size, channel, h*w)
                attn_in = self.cross_attn_norms[i](out)
                attn_in = out.transpose(1,2)
                context_in = self.context_proj[i](context)
                attn_out = self.attention[i](attn_in, context_in, context_in)
                attn_out = attn_out.transpose(1,2).reshape(batch_size, channel, h, w)
                out = out + attn_out
        
        out = self.down_sample_layer(out)
        return out

class MidBlock(nn.Module):
    def __init__(self):
        super().__init__()
