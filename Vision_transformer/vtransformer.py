import torch
from torch import nn
from encoder import Encoder

class VisionTransformer(nn.Module):
    """
        Vision Tranformer implementation from https://arxiv.org/pdf/2010.11929
    """
    
    def __init__(self, 
        image_size,
        patch_size,
        num_layers,
        num_heads,
        embedding_dim,
        forward_dim,
        dropout,
        attention_dropout,
        num_classes,
        representation_size
    ):
        # ex. 224 % 16, 32 % 4
        torch._assert(image_size % patch_size == 0, "image size should be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.forward_dim = forward_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        
        # TODO: add conv_stem_config
        # using https://arxiv.org/abs/2106.14881: Early Conv
        
        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # N = HW/p**2
        seq_len = self.image_size * self.image_size / self.patch_size ** 2
        
        # class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        seq_len += 1
        
        self.encoder = Encoder(
            
        ) 
        
        self.seq_len = seq_len