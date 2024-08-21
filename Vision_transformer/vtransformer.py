import math
import torch
from torch import nn
from encoder import Encoder
from collections import OrderedDict

class SimpleVisionTransformer(nn.Module):
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
        super().__init__()
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
        seq_len = self.image_size * self.image_size // self.patch_size ** 2
        
        # class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        seq_len += 1
        
        self.encoder = Encoder(
            seq_len,
            embedding_dim,
            forward_dim,
            num_layers,
            num_heads,
            attention_dropout,
            dropout
        ) 
        
        self.seq_len = seq_len
        
        head_layers = OrderedDict()
        if representation_size is None:
            head_layers["head"] = nn.Linear(embedding_dim, num_classes)
        else:
            head_layers["pre_logits"] = nn.Linear(embedding_dim, representation_size)
            head_layers["activation"] = nn.Tanh()
            head_layers["head"] = nn.Linear(representation_size, num_classes)
        
        self.heads = nn.Sequential(head_layers)
        
        if isinstance(self.conv_proj, nn.Conv2d):
            # Xavier like initialization
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1/fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        
        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1/fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)
            
        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x):
        N, C, H, W = x.shape
        
        n_h = H // self.patch_size
        n_w = W // self.patch_size
        
        # shape - n emb_dim n_h, n_w
        x = self.conv_proj(x)
        # print(x.shape)
        x = x.reshape(N, self.embedding_dim, n_h * n_w, )
        
        
        # N, seq_len, emb_dim as expected by tranformer
        x = x.permute(0, 2, 1)
        
        return x
    
    def forward(self, x):
        # (N, S, E)
        x = self._process_input(x)
        N = x.shape[0]
        
        batch_class_token = self.class_token.expand(N, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        
        x = x[:, 0]
        x = self.heads(x)
        
        return x
        
