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
        self.initialize()
        
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
        
        if in_channel != out_channel:
            self.down_sample = nn.Conv2d(in_channel, out_channel, 1,1)
        else:
            self.down_sample = nn.Identity()
            
        if attn:
            self.attn = Attn(out_channel)
        else:
            self.attn = nn.Identity
        
        self.initialize()
        
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)
        
    def forward(self, x, temb):
        out = self.block1(x)
        out += self.temb_proj(temb)
        out = self.block2(out)
        
        out = out + self.down_sample(x)
        out = self.attn(out)
        return out
            
class Attn():
    def __init__(
        self,
        dim
    ):
        # self.qkv = nn.Conv2d(dim, dim*3, 1, 1)
        self.q = nn.Conv2d(dim, dim, 1, stride=1, padding=0)
        self.k = nn.Conv2d(dim, dim, 1, stride=1, padding=0)
        self.v = nn.Conv2d(dim, dim, 1, stride=1, padding=0)
        self.merge = nn.Conv2d(dim, dim, 1, 1)
        self.norm = nn.GroupNorm(32, dim)
        self.initialize()
        
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.merge.weight, gain=1e-5)
        
    def forward(self, x):
        # qkv = self.qkv(x)
        # q, k, v  = torch.chunk(qkv, 3, 1)
        N, C, H, W = x.shape
        
        out = self.norm(x)
        q, k, v = self.q(out), self.k(out), self.v(out)
        
        q = q.permute(0,2,3,1)
        k = k.reshape(N, C, H*W)
        
        q *= (C ** -0.5)
        qk = torch.bmm(q, k)
        w = torch.softmax(qk, dim=-1)
        
        v = v.permute(0,2,3,1).reshape(N, H*W, C)
        out = torch.bmm(w, v)
        out = out.reshape(N, H, W, C).permute(0,3,1,2)
        out = self.merge(out)
        
        return x + out

class DownSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down = nn.Conv2d(dim, dim, 3,2,1)
        
    def initialize(self):
        nn. init.xavier_uniform_(self.down.weight)
        nn. init.zeros_(self.down.bias)

    def forward(self, x):
        x = self.main(x)
        return x

class UpSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
        
    def initialize(self):
        nn. init.xavier_uniform_(self.up.weight)
        nn. init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(x)
    
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
                chs.append(curr_channel)
        if i < len(channel_mult) -1:
            self.downblocks.append(DownSample(curr_channel))
            chs.append(curr_channel)        

        self.middle_blocks = nn.ModuleList([
            ResBlock(curr_channel, curr_channel, tdim, dropout, attn=True),
            ResBlock(curr_channel, curr_channel, tdim, dropout, attn=False)
        ])
        
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_channel = channels * mult
            for _ in range(n_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    curr_channel + chs.pop(), 
                    out_channel, 
                    tdim, 
                    dropout, 
                    attn=(i in attn)))
                curr_channel = out_channel
            if i > 0:
                self.upblocks.append(UpSample(curr_channel))
                
        self.tail = nn.Sequential(
            nn.GroupNorm(32, curr_channel),
            nn.SiLU(),
            nn.Conv2d(curr_channel, 3, 3, 1, 1)
        )
        self.initialize()
        
    def initialize(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        nn.init.zeros_(self.tail[-1].bias)
    
    def forward(self, x):
        temb = self.time_embedding(x)
        
        out = self.head(x)
        outs = [out]
        for block in self.downblocks:
            out = block(out, temb)
            outs.append(out)

        out = self.middle_blocks(out)
        
        for block in self.upblocks:
            if isinstance(block, ResBlock):
                out = torch.concat([block, outs.pop()], dim=1)
            out = block(out, temb)
            
        out = self.tail(out)
        
        return out
    
    
if __name__ == "__main__":
    te = TimeEmbedding(3, 6, 12)
    print(te.emb)
    # print(te(torch.tensor([0, 1])))
    # print(te(torch.tensor([0])))