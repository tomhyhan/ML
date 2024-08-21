from torch import nn

class MLP(nn.Module):
    
    def __init__(self, embedding_dim, forward_dim):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, forward_dim),
            nn.GELU(),
            nn.Linear(forward_dim, embedding_dim)
        )
        
    def forward(self, x):
        out = x.clone()
        out = self.mlp(out)
        return out