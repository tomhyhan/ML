import sys
sys.path.append("../")

import torch
from torch.nn import Conv2d, Parameter
from src.utils.weight_init import kaiming_init

class Conv:
    """
        2d Convolutional neural network. Implemented using torch's conv2d for faster computation.
    """
    def __init__(self, C_out, C_in, k, stride=1, padding=0, groups=1,weight_scale=True, relu=True, device="cpu", dtype=torch.float32):
        """
            Initialization of Conv layer:
                weight (filter): (C_out, C_in, k, k) if weight_scalse is set to True, initializa weight with kaiming weight initialization. Otherwise, initialize normally with torch.radn.
                bias: (C_out)
                stride: steps for each conv
                padding: extra padding
                groups: number of conv layer groups 
                params: containing weights and biases
                grads: containing gradients for weights and biases
        """
        if weight_scale:
            self.w = kaiming_init(D_out=C_out, D_in=C_in, k=k, relu=relu, device=device, dtype=dtype)
        else:
            self.w = torch.randn(C_out, C_in, k, k, device=device, dtype=dtype)
        self.b = torch.randn(C_out, device=device, dtype=dtype)    
        
        self.device = device
        self.dtype = dtype
        self.stride = stride
        self.padding= padding
        self.groups = groups
        
        
        self.dw = self.db = None
        self.params = {
            'w' : self.w,
            'b' : self.b,
        }
        self.grads = {}
        self.config = {}
    
    def forward(self, X):
        """
            Compute forward pass for Conv layer.
            
            Input:
                X: (N, C_in, H_in, W_in) input data
            output:
                out: (N, C_out, H_out, W_out) the result of convolution of input data using filter
        """
        self.prev_x = X.clone()
        N, C_in, H_in, W_in = X.shape
        C_out, C_in, k, k = self.w.shape
        conv = Conv2d(C_in, C_out, (k,k), stride=self.stride, padding=self.padding, device=self.device, dtype=self.dtype, groups=self.groups)
        
        conv.weight = Parameter(self.w)
        conv.bias = Parameter(self.b)
        
        self.tx = X.detach()
        self.tx.requires_grad = True
        self.conv = conv
        
        out = self.conv(self.tx)
        self.out = out
        
        return out
        
    
    def backward(self, dout):
        """
            Computes backpropagation for convolutional layer

            Input:
                dout: (N, C_out, H_out, W_out) upstream gradients
            Ouputs:
                dx: (N, C_in, H_in, W_in) downstream gradients w.r.t. x
            Grads:
                dw: (C_out, C_in, k, k) gradients w.r.t w
                db: (C_out) gradients w.r.t b
        """
        
        self.out.backward(dout)
        dx = self.tx.grad.detach()
        dw = self.conv.weight.grad.detach()
        db = self.conv.bias.grad.detach()
        
        self.conv.weight.grad = None
        self.conv.bias.grad = None
        
        self.grads = {
            'w' : dw,
            'b' : db
        }
        
        return dx

    
    def reset(self):
        self.grads = {}