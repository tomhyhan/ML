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
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.normal_(layer.bias, std=1e-6)
                
    def forward(self, x):
        out = x.clone()
        out = self.mlp(out)
        return out