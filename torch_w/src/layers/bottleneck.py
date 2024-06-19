import torch
from src.layers.conv import Conv
from src.layers.bn import BatchNorm
from src.layers.relu import ReLU

class Bottleneck:
    """
        Bottleneck Implementation for Resnet
        
        Conv1x1 - BN - Relu - Conv3x3 - BN - Relu - Conv1x1 - Bn - indentity - relu
    """
    def __init__(self, C_in, C_out, stride, base_width=64, groups=1, device="cpu", dtype=torch.float32, training=True):
        
        self.down_sample = False
        self.expansion = 4
        self.training = training
        
        if stride == 2 or C_in != C_out * self.expansion:
            self.down_sample = True

        width = int(C_out * (base_width / 64.0)) * groups
        
        self.conv1 = Conv(C_in=C_in, C_out=width, k=1, device=device, dtype=dtype)
        self.bn1 = BatchNorm(C=width, device=device, dtype=dtype)
        self.relu1 = ReLU()
        
        self.conv2 = Conv(C_in=width, C_out=width, k=3, stride=stride, padding=1,device=device, dtype=dtype)
        self.bn2 = BatchNorm(C=width, device=device, dtype=dtype)
        self.relu2 = ReLU()

        self.conv3 = Conv(C_in=width, C_out=C_out * self.expansion, k=1, device=device, dtype=dtype)
        self.bn3 = BatchNorm(C=C_out * self.expansion, device=device, dtype=dtype)        
        self.relu3 = ReLU()
        
        self.param_layers = [
            self.conv1, self.bn1, 
            self.conv2, self.bn2, 
            self.conv3, self.bn3, 
        ]
        if self.down_sample:
            self.conv4 = Conv(C_in=C_in, C_out=C_out * self.expansion, k=1, stride=stride , device=device, dtype=dtype)
            self.bn4 = BatchNorm(C=C_out * self.expansion, device=device, dtype=dtype)        
            self.param_layers.append(self.conv4)
            self.param_layers.append(self.bn4)
        


    def forward(self, X):
        """
            Compute forward pass for bottlenect building block
            Inputs:
                X: input data
        """
        identity_X = X.clone()
        
        out = self.conv1.forward(X)
        out = self.bn1.forward(out, training=self.training)
        out = self.relu1.forward(out)
        
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, training=self.training)
        out = self.relu2.forward(out)
        
        if self.down_sample:
            identity_X = self.conv4.forward(identity_X)
            identity_X = self.bn4.forward(identity_X, training=self.training)
        
        out = self.conv3.forward(out)
        out = self.bn3.forward(out, training=self.training)
        
        out += identity_X
        
        out = self.relu3.forward(out)
        
        return out
    
    def backward(self, dout):
        """
            Computes back propagation for Bottlenect block
        """
        dout = self.relu3.backward(dout)
        didentity = dout.clone()
        
        if self.down_sample:
            didentity = self.bn4.backward(didentity)
            didentity = self.conv4.backward(didentity)

        dout = self.bn3.backward(dout)
        dout = self.conv3.backward(dout)
        
        dout = self.relu2.backward(dout)
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)
        
        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
        
        dx = dout + didentity
        
        return dx