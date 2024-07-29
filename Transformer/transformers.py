import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder
from mask import get_subsequent_mask

class Transformers(nn.Module):
    """
        Transformer model
    """
    def __init__(self, num_heads, emp_dim, feedforward_dim, dropout, num_enc_layer, num_dec_layer, vocab_len, device="cpu", dtype=torch.float32):
        """
            implements transformer model with encoder with decoder
        """
        super().__init__()
        
        self.emb_layer = nn.Embedding(vocab_len, emp_dim, device=device, dtype=dtype)
        self.encoder = Encoder(num_enc_layer, num_heads, emp_dim, feedforward_dim, dropout, device, dtype)
        self.decoder = Decoder(num_dec_layer, num_heads, emp_dim, feedforward_dim, vocab_len, dropout, device, dtype)
        
    def forward(self, ques, ques_pos, ans, ans_pos):
        """
            Implements the forward pass of the transformer 
            
            Inputs:
                ques: (N, K)
                ques_pos: (N, K, M)
                ans: (N, K)
                ans_pos: (N, K, M)
        """
        _, K = ans.shape
        ques_emb = self.emb_layer(ques)
        ques_emb_inp = ques_emb + ques_pos 
        
        ans_emb = self.emb_layer(ans)
        ans_emb_inp = ans_emb[:, :-1] + ans_pos[:, :-1]

        mask = get_subsequent_mask(K-1)

        enc_out = self.encoder(ques_emb_inp)
        dec_out = self.decoder(ans_emb_inp, enc_out, mask)
        
        return dec_out
    
if "__main__" == __name__:
    num_heads = 2
    emp_dim = 4
    feedforward_dim = 16
    batches = 3
    vocab_len = 7
    num_enc_layers = 4
    num_dec_layers = 4
    dropout = 0.0
    
    ques =  torch.randint(0, vocab_len, (batches, 9))
    ques_pos =  torch.randn(batches, 9, emp_dim)
    ans =  torch.randint(0, vocab_len, (batches, 5))
    ans_pos =  torch.randn(batches, 5, emp_dim)

    model = Transformers(num_heads, emp_dim, feedforward_dim, dropout, num_enc_layers, num_dec_layers, vocab_len)
    print(ques)
    y = model(ques.long(), ques_pos, ans.long(), ans_pos)
    
    print(y.shape)
    
