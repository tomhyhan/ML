import torch
from .utils.weight_init import kaiming_initialization
from .layers.conv_batch_relu_pool import SequentialConv
from .layers.fc import FullyConnectedLayer
from .layers.softmax import Softmax

class DeepConvNet:
    """
        Deep convolution neural network. Each layer has following architecture except the final layer:
        
        Conv - BatchNorm - Relu - Maxpool?
        
        final Layer is FullyConnectedLayer to apply softmax.
    """
    
    def __init__(self, input_dim, filters, n_classes, reg, batchnorm, device="cpu", dtype=torch.float):
        """
            initialize weights, biases for each layer.
            
            Weight: (C_out, C_in, HH, WW)
            
            Inputs:
                input_dim: (C_in,H,W) input data dimension 
                filters: nested array containing filter size and maxpool status. [[int, bool]...]
                n_classes: number of final output classes
                reg: regularization strength for l2 reg
        """
        self.params = {}
        self.num_layers = len(filters) + 1
        self.reg = reg
        self.batchnorm = batchnorm
        self.device = device
        self.dtype = dtype
        
        C_in, H, W = input_dim
        k = 3
        n_maxpools = 0
        
        for l in range(1, self.num_layers):
            C_out, is_maxpool = filters[l-1]
            if batchnorm:
                self.params[f"gamma{l}"] = torch.ones(C_out, device=device, dtype=dtype)
                self.params[f"beta{l}"] = torch.zeros(C_out, device=device, dtype=dtype)
            if is_maxpool:
                n_maxpools += 1
            self.params[f"W{l}"] = kaiming_initialization(D_in=C_in, D_out=C_out, k=k)
            self.params[f"B{l}"] = torch.zeros(C_in, device=device, dtype=dtype)

            C_in = C_out

        last_HW = H // (2**n_maxpools)
        L = self.num_layers
        
        if batchnorm:
            self.params[f"gamma{L}"] = torch.ones(C_out, device=device, dtype=dtype)
            self.params[f"beta{L}"] = torch.zeros(C_out, device=device, dtype=dtype)
        self.params[f"W{L}"] = kaiming_initialization(D_in=C_in*last_HW*last_HW, D_out=n_classes, k=k)
        self.params[f"B{L}"] = torch.zeros(n_classes, device=device, dtype=dtype)
        
        self.bn_params = []
        if batchnorm:
            for _ in range(self.num_layers):
                # running mean and std will automatically saved in self.bn_params
                self.bn_params.append([{}])
                
    def loss(self, X, Y=None):
        """
            computes loss and gradients for deep Convolutional layer
            
            Inputs:
                X: train data
                Y: true label for X
            Outpus:
                loss: computed loss for current batch
                gradients: grads for each layer through backpropagation 
        """
        self.caches = {}
        for l in range(1, self.num_layers):
            w = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            if self.batchnorm:
                gamma = self.params[f"gamma{l}"]
                beta = self.params[f"beta{l}"]
                bn_param = self.bn_params[l-1]
                X, cache = SequentialConv.forward(X,w,b,gamma,beta, bn_param)
                self.caches[f"cache{l}"] = cache
                
        L = self.num_layers
        w = self.params[f"W{L}"]
        b = self.params[f"b{L}"]
        X, cache = FullyConnectedLayer.forward(X,w,b)
        self.caches[f"cache{L}"] = cache
        
        scores = Softmax.forward(X)
        
        if Y is None:
            # if y is None, if we trying to compute the accuracy of our model. So, simply return the softmax prob.
            return scores        
        
        self.grads = {}
        loss, dout = Softmax(scores)
        
        w = self.params[f"W{L}"]
        cache = self.caches[f"cache{L}"]
        dout, dw, db = FullyConnectedLayer.backward(dout, cache)
        
        self.grads[f"W{L}"] = dw + self.reg * w 
        self.grads[f"b{L}"] = db 
        loss += 0.5 * self.reg * torch.sum(w*w)
        
        for l in reversed(range(1,self.num_layers)):
            w = self.params[f"W{l}"]
            cache = self.caches[f"cache{l}"]
            dout, dw, db, dgamma, dbeta = SequentialConv.backward(dout, cache)
            self.grads[f"W{l}"] = dw + self.reg * w 
            self.grads[f"b{l}"] = db 
            self.grads[f"gamma{l}"] = dgamma 
            self.grads[f"beta{l}"] = dbeta 
            loss += 0.5 * self.reg * torch.sum(w*w)
            
        return loss, self.grads    
            
        