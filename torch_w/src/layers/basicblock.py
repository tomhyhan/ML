import sys
sys.path.append("../")

import torch
from src.layers.conv import Conv
from src.layers.bn import BatchNorm
from src.layers.relu import ReLU


class BasicBlock:
    """
        Basic building block for Resnet
        
        Conv - BN - Relu - Conv - BN - Identity - Relu
    """
    expansion = 1
        
    def __init__(self, C_in, C_out, stride, device="cpu", dtype=torch.float32, base_width=64, groups=1, training=True):
        """
            initialization of basic block with differemt types of layers
            
            Inputs:
                C_in: input channels
                C_out: output channels
                stride: number of stride
                device: device
                dtype: data type
                training: True or False
        """
        self.down_sample = False
        if stride == 2:
            self.down_sample = True
        self.training = training
        
        self.conv1 = Conv(C_out=C_out, C_in=C_in, k=3, stride=stride, padding=1, device=device, dtype=dtype)
        self.bn1 = BatchNorm(C_out, device=device, dtype=dtype)
        self.relu1 = ReLU()
        
        self.conv2 = Conv(C_out=C_out, C_in=C_out, k=3, padding=1, device=device, dtype=dtype)
        self.bn2 = BatchNorm(C_out, device=device, dtype=dtype)
        self.relu2 = ReLU()
        
        self.param_layers = [
            self.conv1, self.bn1, self.conv2, self.bn2
        ]
        
        if self.down_sample:
            self.conv_down = Conv(C_out=C_out, C_in=C_in, k=1, stride=2, device=device, dtype=dtype)
            self.bn_down = BatchNorm(C_out, device=device, dtype=dtype)
            self.param_layers.append(self.conv_down)
            self.param_layers.append(self.bn_down)
        
    def forward(self, X):
        """
            Compute forward pass for basic block layer
            
            Inputs:
                X: (N, C_in, H, W) data input
            Outpus:
                out: result of processing multiple layers
        """
        identity = X.clone()
        
        out = self.conv1.forward(X) 
        out = self.bn1.forward(out, training=self.training)
        out = self.relu1.forward(out)
        
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, training=self.training)
        
        if self.down_sample:
            identity = self.conv_down.forward(identity)
            identity = self.bn_down.forward(identity, training=self.training)
        
        out += identity
        out = self.relu2.forward(out)
        
        return out
    
    def backward(self, dout):
        """
            Compute back propagation for basic block.
            
            Inputs:
                dout: upstream gradients 
            outputs:
                dx: downstrean gradients
        """
        
        dout = self.relu2.backward(dout)
        
        didentity = dout.clone()
        
        if self.down_sample:
            didentity = self.bn_down.backward(didentity)
            didentity = self.conv_down.backward(didentity)
        
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)
        
        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
            
        dx = dout + didentity
        
        return dx