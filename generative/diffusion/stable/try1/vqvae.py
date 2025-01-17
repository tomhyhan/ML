import torch
from torch import nn
from blocks import DownBlock, UpBlock, MidBlock

class VQVAE(nn.Module):
    def __init__(
        self,
        channels,
        config 
    ):
        super().__init__()
        self.img_channels = channels
        self.z_channels = config['z_channels']
        self.codebook_size = config['codebook_size']
        self.down_channels = config['down_channels']
        self.mid_channels = config['mid_channels']
        self.down_sample = config['down_sample']
        self.attn_down = config['attn_down']
        self.norm_channels = config['norm_channels']
        self.num_heads = config['num_heads']
        self.num_down_layers = config['num_down_layers']
        self.num_mid_layers = config['num_mid_layers']
        self.num_up_layers = config['num_up_layers']
        
        self.up_samples = list(reversed(self.down_sample))
        # encoder
        self.enc_down_conv_in = nn.Conv2d(self.z_channels, self.down_channels[-1])
        
        self.enc_down_layer = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.enc_down_layer.append(
                DownBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i+1],
                    t_emb_dim=None,
                    num_heads=self.num_heads,
                    attn=self.attn_down[i],
                    down_sample=self.down_sample[i],
                    num_layers=self.num_down_layers,
                    norm_channels=self.norm_channels
                )
            )

        self.enc_mid_layer = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.enc_mid_layer.append(
                MidBlock(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i+1],
                    t_emb_dim=None,
                    num_heads=self.num_heads,
                    num_layers=self.num_mid_layers,
                    norm_channels=self.norm_channels
                )
            )
        
        self.enc_down_norm_out = nn.GroupNorm(self.norm_channels, self.mid_channels[-1])
        self.enc_down_conv_out = nn.Conv2d(self.mid_channels[-1], self.z_channels, kernel_size=3, padding=1)
        
        # pre quant conv
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        
        # codebook embbeding
        self.codebook = nn.Embedding(self.codebook_size, self.z_channels)
        
        # post quant conv
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        
        # decoder
        self.dec_conv_in = nn.Conv2d(self.z_channels, self.down_channels[-1], kernel_size=3, padding=1)
        
        self.dec_mid_layer = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mid.append(
                MidBlock(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i-1],
                    t_emb_dim=None,
                    num_layers=self.num_mid_layers,
                    num_heads=self.num_heads,
                    norm_channels=self.norm_channels,
                )
            )
        
        self.dec_up_layer = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_up.append(
                UpBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i-1],
                    t_emb_dim=None,
                    num_layers=self.num_up_layers,
                    num_heads=self.num_heads,
                    attn=self.attn_down[i-1],
                    down_sample=self.up_samples[i-1],
                    norm_channels=self.norm_channels
                )
            )
        
        self.dec_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.dec_conv_out = nn.Conv2d(self.down_channels[0], self.img_channels, kernel_size=3, padding=1)
        
    def quantize(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, H*W, C)
        
        dist = torch.cdist(x, self.codebook.weight[None,:,:].repeat(x.size(0), 1, 1))
        # B, HW
        min_dist_indices = dist.argmax(dim=-1)
        
        quant = torch.index_select(self.codebook.weight, 0, min_dist_indices.view(0))
        
        x = x.reshape(B*H*W, C)

        codebook_loss = torch.mean((quant - x.detach())**2)
        commitment_loass = torch.mean((quant.detach() - x)**2)
        
        losses = {
            "codebook_loss": codebook_loss,
            "commitment_loass": commitment_loass
        }
        
        quant = x + (quant - x).detach()
        quant = quant.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return quant, losses, {}
    
    def encoder(self, x):
        out = x
        out = self.enc_down_conv_in(out)
        for down in self.enc_down_layer:
            out = down(out)
        for mid in self.enc_mid_layer:
            out = mid(out)
        out = self.enc_down_norm_out(out)
        out = nn.SELU()(out)
        out = self.enc_down_conv_out(out)
        out = self.pre_quant_conv(out)
        out, quan_loss, _ = self.quantize(out) 
        return out, quan_loss
    
    def decoder(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.dec_conv_in(out)
        for mid in self.dec_mid_layer:
            out = mid(out)
        for up in self.dec_up_layer:
            out = up(out)
        out = self.dec_norm_out(out)
        nn.SiLU()(out)
        out = self.dec_conv_out(out)
        return out
    
    def forward(self, x):
        z, losses = self.encoder(x)
        out = self.decoder(z)
        return out, z, losses
        

        

