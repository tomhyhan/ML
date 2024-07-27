import torch

def get_subsequent_mask(seq, device="cpu", dtype=torch.float32):
    return torch.triu(
        torch.full((seq, seq), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )
    