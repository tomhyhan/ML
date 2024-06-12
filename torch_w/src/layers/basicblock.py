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
    
    def __init__(self, C_in, C_out, stride, device="cpu", dtype=torch.float32, training=True):
        """
            initialization of basic block. 
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
        
        if self.down_sample:
            self.conv_down = Conv(C_out=C_out, C_in=C_in, k=1, stride=2)
        
    def forward(self, X):
        self.identity = X.clone()
        
        out = self.conv1.forward(X) 
        out = self.bn1.forward(out, training=self.training)
        out = self.relu1.forward(out)
        
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, training=self.training)
        
        if self.down_sample:
            self.identity = 
            pass
        
        out += self.identity
        out = self.relu2(out)
        
        return out
        