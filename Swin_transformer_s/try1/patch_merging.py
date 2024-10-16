import torch
from torch import nn
import torch.nn.functional as F

class PatchMerge(nn.Module):
    def __init__(self, emb_dim, norm_layer):
        
        self.emb_dim = emb_dim
        self.norm = norm_layer(4*emb_dim)
        self.reduction = nn.Linear(4*emb_dim, 2*emb_dim)
    
    def forward(self, x):
        _, H, W, _ = x.shape
        
        x = F.pad((0,0,0,W%2,0,H%2,0,0))
        
        x1 = x[..., 0::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 0::2, :]
        x4 = x[..., 1::2, 1::2, :]
        
        x = torch.cat(x1,x2,x3,x4, dim=-1)
        
        x = self.norm(x)
        x = self.reduction(x)
        return x
    