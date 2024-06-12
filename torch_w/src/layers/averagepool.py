import torch
from torch.nn import AdaptiveAvgPool2d

class AveragePool:
    """
        Global Average Pool Layer
    """
    def __init__(self):
        pass
    
    def forward(self, X):
        """
            Compute forward pass for Global Average Pool Layer
            
            Inputs:
                X: (N, C, H, W) input data
            Output:
                out: (N, C, 1, 1) outputs an input data where the W, H is reduce to 1x1
        """
        avgpool = AdaptiveAvgPool2d((1,1))
        self.tx = X.clone()
        self.tx.requires_grad = True
        
        self.out = avgpool(self.tx)
        
        return self.out
        
    def backward(self, dout):
        """
            Computes back propagation for Global Average Pooling

            Inputs:
                dout: upstream gradients
        """
        
        self.out.backward(dout)
        dx = self.tx.grad.detach()
        
        return dx
        