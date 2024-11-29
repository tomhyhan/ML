import math
import torch
from torch import nn

class TimeEmbedding(nn.Module):
    def __init__(
        self,
        time_steps,
        channels,
        tdim
    ):
        super().__init__()
        emb = torch.arange(0, channels, step=2) / channels * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(time_steps).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        print(emb.shape)
        emb = emb.reshape(time_steps, channels)
        print(emb.shape)
        self.emb = emb
        self.time_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(channels, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim)
        )
        
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, t):
        return self.time_embedding(t)

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        tdim,
        dropout,
        attn
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            nn.SiLU(),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        )
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_channel)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channel),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channel, out_channel, 3,1,1)
        )
        
        
        
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        
class UpSample(nn.Module):
    def __init__(self):
        super().__init__()

class Unet(nn.Module):
    def __init__(
        self,
        time_steps,
        channels,
        channel_mult,
        attn,
        n_res_blocks,
        dropout
    ):
        super().__init__()
        tdim = channels * 4
        self.time_embedding = TimeEmbedding(time_steps, channels, tdim)
        
        self.head = nn.Conv2d(3, channels , kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [channels]
        
        curr_channel = channels
        for i, k in enumerate(channel_mult):
            out_channel = channels * k
            for _ in range(n_res_blocks):
                self.downblocks.append(ResBlock(
                    in_channel=curr_channel,
                    out_channel=out_channel,
                    tdim=tdim,
                    dropout=dropout,
                    attn=(i in attn)
                ))
                curr_channel = out_channel
                chs.append(out_channel)
        if i < len(channel_mult) -1:
            self.downblocks.append(DownSample(curr_channel))
            chs.append(curr_channel)        
    
if __name__ == "__main__":
    te = TimeEmbedding(3, 6, 12)
    print(te.emb)
    # print(te(torch.tensor([0, 1])))
    # print(te(torch.tensor([0])))