from torch import nn

class PatchMerge(nn.Module):
    def __init__(self, emb_dim, norm_layer):
        
        self.emb_dim = emb_dim
        self.norm = norm_layer(4*emb_dim)
        self.reduction = nn.Linear(4*emb_dim, 2*emb_dim)
    
    def forward(self):
        pass