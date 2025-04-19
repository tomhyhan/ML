import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            t_emb_dim,
            down_sample,
            num_heads,
            num_layers,
            attn,
            norm_channels,
            cross_attn=False,
            context_dim=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_emb_dim = t_emb_dim
        self.down_sample = down_sample
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn = attn
        self.norm_channels = norm_channels
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        
        self.resnet_conv1 = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(self.norm_channels, self.in_channels if i == 0 else self.out_channels),
                nn.SiLU(),
                nn.Conv2d(self.in_channels if i == 0 else self.out_channels, out_channels, 3, 1, 1)
            ) for i in range(self.num_layers)
        )
        
        if self.t_emb_dim is not None:
            self.t_emb_layer = nn.ModuleList(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels, 1)
                ) for _ in range(self.num_layers)
            ) 
            
        self.resnet_conv2 = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(self.norm_channels, self.out_channels),
                nn.SiLU(),
                nn.Conv2d(self.out_channels, out_channels, 3, 1, 1)
            ) for _ in range(self.num_layers)
        )
        
        if self.attn:
            self.attn_norms = nn.ModuleList(
                nn.GroupNorm(self.norm_channels, self.out_channels)
                for _ in range(self.num_layers)
            )
            self.attn_layers = nn.ModuleList(
                nn.MultiheadAttention(self.out_channels, self.num_heads, batch_first=True)
                for _ in range(self.num_layers)
            )

        if self.cross_attn:
            self.cross_attn_norms = nn.ModuleList(
                nn.GroupNorm(self.norm_channels, self.out_channels)
                for _ in range(self.num_layers)
            )
            self.cross_attn_layers = nn.ModuleList(
                nn.MultiheadAttention(self.out_channels, self.num_heads, batch_first=True)
                for _ in range(self.num_layers)
            )
            self.context_proj = nn.ModuleList(
                nn.Linear(self.context_dim, self.out_channels)
                for _ in range(self.num_layers)
            )
            
        self.residual_blocks = nn.ModuleList([
            nn.Conv2d(self.in_channels if i == 0 else self.out_channels, self.out_channels, 1)
            for i in range(self.num_layers)
        ])
        
        self.down_sample_block = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.down_sample else nn.Identity()
        
    def forward(self, x, t_emb=None, context=None):
        out = x
        for i in range(self.num_layers):
            residual_input = out
            out = self.resnet_conv1[i](out)
            if t_emb is not None:
                out = out + self.t_emb_layer[i](t_emb)[:, :, None, None]
            out = self.resnet_conv2[i](out)
            out = out + self.residual_blocks(residual_input)
            
            if self.attn:
                B, C, H, W = out.shape
                attn_input = out.reshape(B, C, H*W)
                attn_input = self.attn_norms[i](attn_input)
                attn_input = attn_input.transpose(1, 2)
                attn_output, _ = self.attn_layers[i](attn_input, attn_input, attn_input)
                attn_output = attn_output.transpose(1,2).reshape(B, C, H, W)
                out = out + attn_output
            
            if self.cross_attn:
                B, C, H, W = out.shape
                cross_attn_input = out.reshape(B, C, H*W)
                cross_attn_input = self.cross_attn_norms[i](cross_attn_input)
                cross_attn_input = cross_attn_input.transpose(1, 2)
                proj = self.context_proj[i](context)
                cross_attn_output, _ = self.cross_attn_layers[i](cross_attn_input, proj, proj)
                cross_attn_output = cross_attn_output.transpose(1,2).reshape(B, C, H, W)
                out = out + cross_attn_output
        
        out = self.down_sample_block(out)
        return out
    
class MidBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            t_emb_dim,
            num_heads,
            num_layers,
            norm_channels,
            cross_attn=False,
            context_dim=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_emb_dim = t_emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.norm_channels = norm_channels
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        
        self.resnet_conv1 = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(self.norm_channels, self.in_channels if i == 0 else self.out_channels),
                nn.SiLU(),
                nn.Conv2d(self.in_channels if i == 0 else self.out_channels, out_channels, 3, 1, 1)
            ) for i in range(self.num_layers + 1)
        )
        
        if self.t_emb_dim is not None:
            self.t_emb_layer = nn.ModuleList(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels, 1)
                ) for _ in range(self.num_layers + 1)
            ) 
            
        self.resnet_conv2 = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(self.norm_channels, self.out_channels),
                nn.SiLU(),
                nn.Conv2d(self.out_channels, out_channels, 3, 1, 1)
            ) for _ in range(self.num_layers + 1)
        )
        
        self.attn_norms = nn.ModuleList(
            nn.GroupNorm(self.norm_channels, self.out_channels)
            for _ in range(self.num_layers)
        )
        self.attn_layers = nn.ModuleList(
            nn.MultiheadAttention(self.out_channels, self.num_heads, batch_first=True)
            for _ in range(self.num_layers)
        )

        if self.cross_attn:
            self.cross_attn_norms = nn.ModuleList(
                nn.GroupNorm(self.norm_channels, self.out_channels)
                for _ in range(self.num_layers)
            )
            self.cross_attn_layers = nn.ModuleList(
                nn.MultiheadAttention(self.out_channels, self.num_heads, batch_first=True)
                for _ in range(self.num_layers)
            )
            self.context_proj = nn.ModuleList(
                nn.Linear(self.context_dim, self.out_channels)
                for _ in range(self.num_layers)
            )
            
        self.residual_blocks = nn.ModuleList([
            nn.Conv2d(self.in_channels if i == 0 else self.out_channels, self.out_channels, 1)
            for i in range(self.num_layers)
        ])
                
        
    def forward(self, x, t_emb=None, context=None):
        out = x
        residual_input = out
        out = self.resnet_conv1[0](out)
        if t_emb is not None:
            out = out + self.t_emb_layer[0](t_emb)[:, :, None, None]
        out = self.resnet_conv2[0](out)
        out = out + self.residual_blocks(residual_input)

        for i in range(self.num_layers):
            B, C, H, W = out.shape
            attn_input = out.reshape(B, C, H*W)
            attn_input = self.attn_norms[i](attn_input)
            attn_input = attn_input.transpose(1, 2)
            attn_output, _ = self.attn_layers[i](attn_input, attn_input, attn_input)
            attn_output = attn_output.transpose(1,2).reshape(B, C, H, W)
            out = out + attn_output
            
            if self.cross_attn:
                B, C, H, W = out.shape
                cross_attn_input = out.reshape(B, C, H*W)
                cross_attn_input = self.cross_attn_norms[i](cross_attn_input)
                cross_attn_input = cross_attn_input.transpose(1, 2)
                proj = self.context_proj[i](context)
                cross_attn_output, _ = self.cross_attn_layers[i](cross_attn_input, proj, proj)
                cross_attn_output = cross_attn_output.transpose(1,2).reshape(B, C, H, W)
                out = out + cross_attn_output
                
            residual_input = out
            out = self.resnet_conv1[i+1](out)
            if t_emb is not None:
                out = out + self.t_emb_layer[i+1](t_emb)[:, :, None, None]
            out = self.resnet_conv2[i+1](out)
            out = out + self.residual_blocks(residual_input)
        
        return out
        
        
class UpBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            t_emb_dim,
            up_sample,
            num_heads,
            num_layers,
            attn,
            norm_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_emb_dim = t_emb_dim
        self.up_sample = up_sample
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn = attn
        self.norm_channels = norm_channels
        
        self.resnet_conv1 = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(self.norm_channels, self.in_channels if i == 0 else self.out_channels),
                nn.SiLU(),
                nn.Conv2d(self.in_channels if i == 0 else self.out_channels, out_channels, 3, 1, 1)
            ) for i in range(self.num_layers)
        )
        
        if self.t_emb_dim is not None:
            self.t_emb_layer = nn.ModuleList(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels, 1)
                ) for _ in range(self.num_layers)
            ) 
            
        self.resnet_conv2 = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(self.norm_channels, self.out_channels),
                nn.SiLU(),
                nn.Conv2d(self.out_channels, out_channels, 3, 1, 1)
            ) for _ in range(self.num_layers)
        )
        
        if self.attn:
            self.attn_norms = nn.ModuleList(
                nn.GroupNorm(self.norm_channels, self.out_channels)
                for _ in range(self.num_layers)
            )
            self.attn_layers = nn.ModuleList(
                nn.MultiheadAttention(self.out_channels, self.num_heads, batch_first=True)
                for _ in range(self.num_layers)
            )

            
        self.residual_blocks = nn.ModuleList([
            nn.Conv2d(self.in_channels if i == 0 else self.out_channels, self.out_channels, 1)
            for i in range(self.num_layers)
        ])
        
        self.up_sample_block = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1) if self.up_sample else nn.Identity()
        
    def forward(self, x, out_down=None, t_emb=None):
        out = x 
        out = self.up_sample_block(out)
        
        if out_down is not None:
            out = torch.concat([out, out_down], dim=1)
            
        for i in range(self.num_layers):
            residual_input = out
            out = self.resnet_conv1[i](out)
            if t_emb is not None:
                out = out + self.t_emb_layer[i](t_emb)[:, :, None, None]
            out = self.resnet_conv2[i](out)
            out = out + self.residual_blocks(residual_input)
            
            if self.attn:
                B, C, H, W = out.shape
                attn_input = out.reshape(B, C, H*W)
                attn_input = self.attn_norms[i](attn_input)
                attn_input = attn_input.transpose(1, 2)
                attn_output, _ = self.attn_layers[i](attn_input, attn_input, attn_input)
                attn_output = attn_output.transpose(1,2).reshape(B, C, H, W)
                out = out + attn_output
        
        return out
        