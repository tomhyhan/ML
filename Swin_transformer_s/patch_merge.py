import torch
from torch import nn
import torch.nn.functional as F

class SimplePatchMerging(nn.Module):
    """
        Patch Merging Layer at the end of each Swin Transformer layer
        
        Args:
            dim (int): Number of input channels
            norm_layer (nn.Module): Normalization layer, Default: nn.LayerNorm
    """    

    def __init__(self, dim, norm_layer = nn.LayerNorm):
        super().__init__()
        
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim ,bias=False)
        self.norm = norm_layer(4*dim)
    
    def forward(self, x):
        N, H, W, C = x.shape
        
        x = F.pad(x, (0, 0, 0, W%2, 0, H%2))
        
        x0 = x[:, 0::2, 0::2]
        x1 = x[:, 0::2, 1::2]
        x2 = x[:, 1::2, 0::2]
        x3 = x[:, 1::2, 1::2]
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)

        x = self.norm(x)
        x = self.reduction(x)
        return x
