import torch
import math

def position_encoding_sinusoid(K, M):
    """
    An implementation of the sinousoidal positional encodings.

    args:
        K: int representing sequence length
        M: int representing embedding dimension for the sequence

    return:
        y: a Tensor of shape (1, K, M)

    """

    pe = torch.zeros(K, M)
    position = torch.arange(0, K, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, M, 2).float() * (-math.log(10000.0) / M))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe