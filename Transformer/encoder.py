import torch
from torch import nn
from feedforward import FeedForward

class EncoderBlock(nn.Module):
    """
        Transformers Encoder block 
        
        Architecture:
        
        inp -> multi_head_attention -> out1 -> dropout -> layer_norm(out1 + inp) -> out2 \
    -> feedforward -> out3 -> dropout -> layer_norm(out3 + out2) -> out
    
    """
    def __init__(self, num_heads, emp_dim, feedforward_dim, dropout, device="cpu", dtype=torch.float32):
        super().__init__()
        
        self.mutiheadattn = nn.MultiheadAttention(emp_dim, num_heads, batch_first=True, device=device, dtype=dtype)
        self.feed_forward = FeedForward(emp_dim, feedforward_dim, device=device, dtype=dtype)
        self.norm1 = nn.LayerNorm(emp_dim, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(emp_dim, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x):
        """
            The implementation of the forward pass of the encorder block of the Tranformer model
            
            N : number of batches
            K : number of sequences
            N : number of q,k,v dims
            
            input:
                x: (N, K, M) 
        """
        out1, _ = self.mutiheadattn(x, x, x)
        out1 = self.dropout(out1)
        out2 = self.norm1(out1 + x)
        out3 = self.feed_forward(out2)
        out3 = self.dropout(out3)
        y = self.norm2(out3 + out2)
        
        return y

class Encoder(nn.Module):
    """
        Transformers Encoder  
    
    """
    def __init__(self, num_layers, num_heads, emp_dim, feedforward_dim, dropout, device="cpu", dtype=torch.float32):
        super().__init__()
        
        
        self.layers = nn.ModuleList([
            EncoderBlock(num_heads, emp_dim, feedforward_dim, dropout, device, dtype) 
            for _ in range(num_layers)
        ])

    def forward(self, enc_inp):
        """
            The implementation of the forward pass of the encoder of the Tranformer model
        """
        out = enc_inp.clone()
        for encoder in self.layers:
            out = encoder(out)
        return out