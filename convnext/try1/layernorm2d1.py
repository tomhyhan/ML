from torch import Tensor
from torch import nn
import torch.nn.functional as F

class LayerNorm2d1(nn.LayerNorm):
    def forward(self, x: Tensor):
        x = x.permute(0,2,3,1)
        x = F.layer_norm(x, normalized_shape=self.normalized_shape, weight=self.weight, bias=self.bias)
        x = x.permute(0,3,1,2)
