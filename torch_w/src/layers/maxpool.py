import torch
from torch.nn import MaxPool2d, Parameter


class Maxpool:
    """
        Maxpool neural network layer
    """
    
    def __init__(self, k=2, stride=2, padding=0) -> None:
        """
            initialize maxpool layer
            
            Input:
                k: H W of maxpool filter
                stride
                padding
        """
        self.k = k
        self.stride = stride
        self.padding = padding
        
    def forward(self, X):
        """
            Computes forward pass for maxpool layer

            Inputs:
                X: input data
        """
        self.prev_x = X
        
        self.tx = X.clone()
        self.tx.requires_grad = True
        
        self.maxpool = MaxPool2d(self.k, self.stride, self.padding)

        self.out = self.maxpool(self.tx)

        return self.out
    
    def backward(self, dout):
        """
            Computes backward pass for maxpool layer
            
            Inputs:
                dout: upstream gradients
        """
        self.out.backward(dout)
        dx = self.tx.grad.detach()
        return dx
        
        