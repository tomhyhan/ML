import torch
from torch import nn
from feedforward import FeedForward

class DecoderBlock(nn.Module):
    """
        Transformers Decoder block 
        
        Architecture:
        
        inp -> masked_multi_head_attention -> dropout -> layer_norm(inp + out1) -> out2 
    -> multi_head_attention (with encoder output) -> dropout -> layer_norm(out2 + out3) -> out4 
    -> feed_forward -> dropout -> layer_norm(out4 + out5) -> out
    """
    def __init__(self, num_heads, emp_dim, feedforward_dim, dropout,  device="cpu", dtype=torch.float32):
        super().__init__()
        
        self.mutiheadattn = nn.MultiheadAttention(emp_dim, num_heads, batch_first=True, device=device, dtype=dtype)
        self.crossattn= nn.MultiheadAttention(emp_dim, num_heads, batch_first=True, device=device, dtype=dtype)
        self.feed_forward = FeedForward(emp_dim, feedforward_dim, device=device, dtype=dtype)
        self.norm1 = nn.LayerNorm(emp_dim, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(emp_dim, device=device, dtype=dtype)
        self.norm3 = nn.LayerNorm(emp_dim, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, dec_inp, enc_inp, mask):
        """
            The implementation of the forward pass of the encorder block of the Tranformer model
            
            N : number of batches
            K : number of sequences
            N : number of q,k,v dims
            
            input:
                dec_inp: (N, K, M) 
                enc_inp: (N, K, M) 
                mask: (K, K) 

        """
        out1, _ = self.mutiheadattn(dec_inp, dec_inp, dec_inp, attn_mask=mask)
        out1 = self.dropout(out1)
        out2 = self.norm1(out1 + dec_inp)
        
        out3, _ = self.crossattn(out2, enc_inp, enc_inp)
        out3 = self.dropout(out3)
        out4 = self.norm2(out2 + out3)
        
        out5 = self.feed_forward(out4)
        out5 = self.dropout(out5)
        y = self.norm3(out4 + out5)

        return y
    
class Decoder(nn.Module):
    """
        Transformers Decoder  
    
    """
    def __init__(self, num_layers, num_heads, emp_dim, feedforward_dim, vocab_len, dropout=0, device="cpu", dtype=torch.float32):
        super().__init__()
        
        
        self.layers = nn.ModuleList([
            DecoderBlock(num_heads, emp_dim, feedforward_dim, dropout, device, dtype) 
            for _ in range(num_layers)
        ])
        self.proj_to_vocab = nn.Linear(emp_dim, vocab_len)
        nn.init.xavier_uniform_(self.proj_to_vocab.weight) 

    def forward(self, dec_inp, enc_inp, mask=None):
        """
            The implementation of the forward pass of the decoder of the Tranformer model
        """
        out = dec_inp.clone()
        for decoder in self.layers:
            out = decoder(out, enc_inp, mask)
        out = self.proj_to_vocab(out)
        return out