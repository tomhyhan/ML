import sys
sys.path.append("../")

import torch
from src.utils.weight_init import kaiming_init

class FullyConnectedLayer:
    """
        Fully connected neural network         
    """
    def __init__(self, D, M, relu=True, weight_scale=True, device="cpu", dtype=torch.float32):
        """
            Initialize the weight and biases for fc layer.

            D - input dimension, M - output dimension
            self.w: (D, M) demension. use kaiming regularization to initialize weights when weight scale is True. 
            self.biases: (M) demension
        """
        if weight_scale:
            self.w = kaiming_init(D_in=D, D_out=M, relu=relu, device=device, dtype=dtype)
        else:
            self.w = torch.randn(D, M, device=device, dtype=dtype)
            
        self.b = torch.randn(M, device=device, dtype=dtype)
        
        # fix: refactor this with module super class
        self.params = {
            'w' : self.w,
            'b' : self.b,
        }
        self.configs = {
            'w' : {},
            'b' : {},
        }
        self.grads = {}

                
    def forward(self, X):
        """
            Compute forward pass for the fully connected layer. 
            
            Inputs:
                X: (N, D) input data
            Outpus:
                out: (N, M)
        """
        N = X.shape[0]
        self.prev_x = X
        out = torch.matmul(X.reshape(N, -1), self.w) + self.b

        return out 
    
    def backward(self, dup):
        """
            Compute back propagation for fc layer.
            
            X: (N, D)
            w: (D, M)
            biases: (M)
            
            Inputs:
                dup: (N, M) upstream gradients
            Outputs:
                dout: (N, D) downstream gradients w.r.t x
            Grad parameter:
                dw: (D, M) gradients w.r.t. w
                db: (M, ) gradients w.r.t. b
        """
        N = self.prev_x.shape[0]
        db = torch.sum(dup, dim=0)
        dw = torch.matmul(self.prev_x.reshape(N, -1).T, dup) 
        dout = torch.matmul(dup, self.w.T)

        self.grads['w'] = dw
        self.grads['b'] = db
        
        return dout.reshape(self.prev_x.shape)
    
    def reset_grads(self):
        """
            Reset gradients
        """
        self.grads = {}
    
    
