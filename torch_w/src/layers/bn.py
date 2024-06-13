import torch
from torch.nn import BatchNorm2d, Parameter

class BatchNorm:
    """
        Batchnorm neural network
    """
    def __init__(self, C, device="cpu", dtype=torch.float32):
        """
            Initialize batch norm parameters 
        """
        self.w = torch.ones(C, device=device, dtype=dtype)
        self.b = torch.zeros(C, device=device, dtype=dtype)
        self.running_var = torch.ones(C, device=device, dtype=dtype)
        self.running_mean = torch.zeros(C, device=device, dtype=dtype)
        self.device = device
        self.dtype= dtype
        self.C = C
        
        self.dw = self.db = None
        
        self.params = {
            'w': self.w,
            'b': self.b
        }
        self.config = {}
        self.grads = {}
    
    def forward(self, X, training=True):
        """
            Computes forward pass for batchnorm layer
            
            Inputs:
                X: (N, C, H, W) input data
                training: training or testing
            Outputs:
                out: (N, C, H, W) normalized output
        """        
        self.prev_x = X
        
        self.bn = BatchNorm2d(num_features=self.C, device=self.device, dtype=self.dtype)
        
        if training:
            self.bn.train()
        else:
            self.bn.eval()
        
        self.tx = X.detach()
        self.tx.requires_grad = True
        
        self.bn.weight = Parameter(self.w)
        self.bn.bias = Parameter(self.b)
        self.bn.running_mean = Parameter(self.running_mean, requires_grad=False)
        self.bn.running_var = Parameter(self.running_var, requires_grad=False)
        self.out = self.bn(self.tx)
        
        return self.out

    def backward(self, dout):
        """
            Computes back propagation for batch norm layer
            
            Inputs:
                dout: (N, C, H, W) upstream gradients
            Outputs:
                dx: (N, C, H, W) gradients w.r.t x
            Grads:
                dw: (C) gradients w.r.t. gamma
                db: (C) gradietns w.r.t. beta
        """
        self.out.backward(dout, retain_graph=True)
        
        dx = self.tx.grad.detach()
        dw = self.bn.weight.grad.detach()
        db = self.bn.bias.grad.detach()
        
        self.bn.weight.grad = None
        self.bn.bias.grad = None
        
        self.grads['w'] = dw
        self.grads['b'] = db
        
        return dx
    
    def reset_grads(self):
        self.grads = {}
    
    def __repr__(self):
        return __name__