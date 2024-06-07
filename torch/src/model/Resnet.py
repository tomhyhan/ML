import sys
sys.path.append("../")

import torch
from src.utils.weight_init import kaiming_initialization
from src.layers.basicblock import BasicBlock
from src.layers.averagepool import AveragePool
from src.layers.fc import FullyConnectedLayer
from src.layers.softmax import Softmax
from src.utils.weight_init import kaiming_initialization

# 1. add one conv layer before starting to increase the C size
# 2. just start with block layers 

class ResNet:
    """
        Residual Network. 
        
        Following Building blocks are needed:
            - conv3x3: 
                kernel size = 3
                padding = 1
                stride = 1
            - conv1x1
                kernelsize = 1
                padding = 0
                stride = 1
            - Basic Block
                conv3x3 - bn - relu - conv3x3 - bn - identity - relu
            - Bottle neck (may requires different resnet model)
                conv1x1 - bn - relu - conv3x3 - bn - relu
                - conv1x1 - bn - identity - relu
                 
        Each layer has following architecture except the final layer:
        
        Basic Conv3 - BatchNorm - Relu - Maxpool?
        
        final Layer is FullyConnectedLayer to apply softmax.
    """
    
    def __init__(self, input_dim, layers, n_classes, reg, device="cpu", dtype=torch.float):
        """
            initialize weights, biases for each layer.
            
            Weight: (C_out, C_in, HH, WW)
            
            Inputs:
                input_dim: (C_in,H,W) input data dimension 
                layers: array containing number of blocks and channel size for each layer. 
                n_classes: number of final output classes
                reg: regularization strength for l2 reg
        """
        self.params = {}
        self.bn_params = {}
        
        self.reg = reg
        self.device = device
        self.dtype = dtype
        C_in, H, W = input_dim

        # layers [1,1,1,1]
        # channels [8,16,32,64] 
        self.first_blocks = []
        self.n_blocks = 1
        for layer, C_out in layers:
            weights, biases, gammas, betas, bn_params = self.init_first_block_params(C_out, C_in)
            C_in = C_out
            self.params[f"W{self.n_blocks}"] = weights 
            self.params[f"b{self.n_blocks}"] = biases
            self.params[f"gamma{self.n_blocks}"] = gammas 
            self.params[f"beta{self.n_blocks}"] = betas 
            self.bn_params[f"bn_params{self.n_blocks}"] = bn_params 
            self.first_blocks.append(self.n_blocks)
            self.n_blocks += 1
            for _ in range(1, layer):
                weights, biases, gammas, betas, bn_params = self.init_block_params(C_out, )
                self.params[f"W{self.n_blocks}"] = weights 
                self.params[f"b{self.n_blocks}"] = biases
                self.params[f"gamma{self.n_blocks}"] = gammas 
                self.params[f"beta{self.n_blocks}"] = betas 
                self.bn_params[f"bn_params{self.n_blocks}"] = bn_params 
                self.n_blocks += 1

        L = self.n_blocks + 1        
        self.params[f"W{L}"] = kaiming_initialization(D_in=C_out, D_out=n_classes, relu=False,device=device, dtype=dtype)
        self.params[f"b{L}"] = torch.zeros(n_classes, device=device, dtype=dtype)
                
    def init_block_params(self, C_out):
        weights = [
            kaiming_initialization(D_in=C_out, D_out=C_out, k=3, device=self.device, dtype=self.dtype),
            kaiming_initialization(D_in=C_out, D_out=C_out, k=3, device=self.device, dtype=self.dtype),
        ]
        biases = [
            torch.zeros(C_out, device=self.device, dtype=self.dtype),
            torch.zeros(C_out, device=self.device, dtype=self.dtype),
        ]
        gammas = [
            torch.ones(C_out, device=self.device, dtype=self.dtype),
            torch.ones(C_out, device=self.device, dtype=self.dtype),
        ]
        betas = [
            torch.zeros(C_out, device=self.device, dtype=self.dtype),
            torch.zeros(C_out, device=self.device, dtype=self.dtype),
        ]
        bn_params = [{},{}]
        
        return weights, biases, gammas, betas, bn_params
    
    
    def init_first_block_params(self, C_out, C_in):
        weights = []
        biases = []
        gammas = []
        betas = []
        
        weights = [
            kaiming_initialization(D_in=C_in, D_out=C_out, k=3, device=self.device, dtype=self.dtype),
            kaiming_initialization(D_in=C_out, D_out=C_out, k=3, device=self.device, dtype=self.dtype),
            kaiming_initialization(D_in=C_in, D_out=C_out, k=1, device=self.device, dtype=self.dtype),
        ]
        biases = [
            torch.zeros(C_out, device=self.device, dtype=self.dtype),
            torch.zeros(C_out, device=self.device, dtype=self.dtype),
            torch.zeros(C_out, device=self.device, dtype=self.dtype),
        ]
        gammas = [
            torch.ones(C_out, device=self.device, dtype=self.dtype),
            torch.ones(C_out, device=self.device, dtype=self.dtype),
            torch.ones(C_out, device=self.device, dtype=self.dtype),
        ]
        betas = [
            torch.zeros(C_out, device=self.device, dtype=self.dtype),
            torch.zeros(C_out, device=self.device, dtype=self.dtype),
            torch.zeros(C_out, device=self.device, dtype=self.dtype),
        ]
        bn_params = [{},{},{}]
        
        return weights, biases, gammas, betas, bn_params
        
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
        for l in range(1, self.n_blocks):
            weights = self.params[f"W{l}"]
            biases = self.params[f"b{l}"]
            gammas = self.params[f"gamma{l}"]
            betas = self.params[f"beta{l}"]
            bn_params = self.bn_params[f"bn_params{l}"]
            if l in self.first_blocks:
                stride = 2
                down_sample = True
            else:
                stride = 1
                down_sample = False
            X, cache = BasicBlock.forward(X, weights, biases, gammas, betas, bn_params, stride, down_sample)
            self.caches[f"cache{l}"] = cache
            
        X, cache_avg = AveragePool.forward(X)
        self.caches[f"cache_avg"] = cache_avg
        
        L = self.n_blocks + 1
        w = self.params[f"W{L}"]
        b = self.params[f"b{L}"]
        
        X, cache_fc = FullyConnectedLayer.forward(X,w,b)
        self.caches[f"cache_fc"] = cache_fc
        
        scores = Softmax.forward(X)
        
        if Y is None:
            # if y is None, if we trying to compute the accuracy of our model. So, simply return the softmax prob.
            return scores        
        
        self.grads = {}
        loss, dout = Softmax.backward(scores, Y)
        w = self.params[f"W{L}"]
        loss += 0.5 * self.reg * torch.sum(w*w)
        
        cache_fc = self.caches[f"cache_fc"]
        
        dout, dw, db = FullyConnectedLayer.backward(dout, cache_fc)
        
        self.grads[f"W{L}"] = dw + self.reg * w 
        self.grads[f"b{L}"] = db 

        cache_avg = self.caches["cache_avg"]
        dout = AveragePool.backward(dout,cache_avg)

        for l in reversed(range(1,self.n_blocks)):
            weights = self.params[f"W{l}"]
            for w in weights:
                loss += 0.5 * self.reg * torch.sum(w*w)
            cache = self.caches[f"cache{l}"]
            dout, dws, dbs, dgammas, dbetas= BasicBlock.backward(dout, cache)
            self.grads[f"W{l}"] = [ dw + self.reg * w for dw, w in zip(dws, weights)] 
            self.grads[f"b{l}"] = dbs 
            self.grads[f"gamma{l}"] = dgammas 
            self.grads[f"beta{l}"] = dbetas 
            
        return loss, self.grads    
            
        