from torch import nn
import  torch.nn.functional as F
from torch import Tensor

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor):
        x = x.permute(0,2,3,1)
        x = F.layer_norm(self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0,3,1,2)
        return x
