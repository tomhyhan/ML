import torch
from torch import nn
from mlp import MLP
from torchvision.models.vision_transformer import MLPBlock

class EncoderBlock(nn.Module):
    def __init__(self,
        embedding_dim,
        forward_dim,
        num_heads,
        attention_dropout,
        dropout,
    ):
        super().__init__()
        # norm - multihead layer - dropout - 
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=attention_dropout, batch_first=True)
        
        # norm - mlp layer - dropout - 
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.mlp = MLP(embedding_dim, forward_dim)
        # self.mlp = MLPBlock(embedding_dim, forward_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = x.clone()
        
        out = self.norm1(out)
        out, _ = self.attn(out, out, out, need_weights=False)
        out = self.dropout(out)
        out1 = out + x
        
        out1 = self.norm2(out1)
        out1 = self.mlp(out1)
        out2 = self.dropout(out1)
        
        return out2 + out1        

class Encoder(nn.Module):
    
    def __init__(self,
        seq_len,
        embedding_dim,
        forward_dim,
        num_layers,
        num_heads,
        attention_dropout,
        dropout,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_len, embedding_dim).normal_(std=0.02))
        
        self.layers = nn.ModuleList([
            EncoderBlock(embedding_dim, forward_dim, num_heads, attention_dropout, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        out = x.clone()
        out += self.pos_embedding
        for layer in self.layers:
            out = layer(out)
        out = self.norm(out)
        return out
    
# if __name__ == "__main__":
#     encoder = Encoder(10, 32, 64, 4, 4, 0.5, 0.5)
#     print(encoder.test.weight.shape)
#     print(encoder.test)
#     print(encoder.pos_embedding.shape)