import sys
sys.path.append("../")

import torch
from src.utils.weight_init import kaiming_initialization
from src.layers.conv_batch_relu_pool import SequentialConv
from src.layers.fc import FullyConnectedLayer
from src.layers.softmax import Softmax

class DeepConvNet:
    """
        Deep convolution neural network. Each layer has following architecture except the final layer:
        
        Conv - BatchNorm - Relu - Maxpool?
        
        final Layer is FullyConnectedLayer to apply softmax.
    """
    
    def __init__(self, input_dim,filters, n_classes, reg, batchnorm, weight_scale=1e-3, device="cpu", dtype=torch.float):
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
        self.filters = filters
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
                
            if weight_scale == "kaiming":
                self.params[f"W{l}"] = kaiming_initialization(D_in=C_in, D_out=C_out, k=k, device=device, dtype=dtype)
            else:
                self.params[f"W{l}"] = torch.randn(C_out, C_in, k, k, device=device, dtype=dtype) * weight_scale
                
            self.params[f"b{l}"] = torch.zeros(C_out, device=device, dtype=dtype)
            # print("init", self.params[f"W{l}"].sum())
            C_in = C_out

        last_HW = H // (2**n_maxpools)
        L = self.num_layers
        
        # if batchnorm:
        #     self.params[f"gamma{L}"] = torch.ones(C_out, device=device, dtype=dtype)
        #     self.params[f"beta{L}"] = torch.zeros(C_out, device=device, dtype=dtype)
        
        if weight_scale == "kaiming":
            self.params[f"W{L}"] = kaiming_initialization(D_in=C_in*last_HW*last_HW, D_out=n_classes, relu=False,device=device, dtype=dtype)
        else:
            self.params[f"W{L}"] = torch.randn(C_in*last_HW*last_HW, n_classes, device=device, dtype=dtype) * weight_scale
            
        self.params[f"b{L}"] = torch.zeros(n_classes, device=device, dtype=dtype)
        
        self.bn_params = []
        if batchnorm:
            for _ in range(self.num_layers-1):
                # running mean and std will automatically saved in self.bn_params
                self.bn_params.append({})
    
    def save(self, file_path):
        checkpoint = {
            "params": self.params,
            "dtype": self.dtype,
            "reg": self.reg,
            "batchnorm": self.batchnorm,
            "bn_params": self.bn_params,
            "filters": self.filters,
        }
        torch.save(checkpoint, file_path)

    def load(self, file_path, dtype, device):
        checkpoint = torch.load(file_path, map_location="cpu")
        self.reg = checkpoint["reg"]
        self.batchnorm = checkpoint["batchnorm"] 
        self.bn_params = checkpoint["bn_params"] 
        self.filters = checkpoint["filters"] 
        self.params = checkpoint["params"] 
        self.dtype = dtype
        self.device = device
        
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        
        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)
        

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
                # print("bn_param", bn_param)
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
        loss, dout = Softmax.backward(scores, Y)
        w = self.params[f"W{L}"]
        loss += 0.5 * self.reg * torch.sum(w*w)
        cache = self.caches[f"cache{L}"]
        dout, dw, db = FullyConnectedLayer.backward(dout, cache)
        
        self.grads[f"W{L}"] = dw + self.reg * w 
        self.grads[f"b{L}"] = db 
        
        for l in reversed(range(1,self.num_layers)):
            w = self.params[f"W{l}"]
            # print(l)
            # print("inside:", torch.sum(w*w))
            loss += 0.5 * self.reg * torch.sum(w*w)
            cache = self.caches[f"cache{l}"]
            dout, dw, db, dgamma, dbeta = SequentialConv.backward(dout, cache)
            self.grads[f"W{l}"] = dw + self.reg * w 
            self.grads[f"b{l}"] = db 
            self.grads[f"gamma{l}"] = dgamma 
            self.grads[f"beta{l}"] = dbeta 
            
        return loss, self.grads    
            
        