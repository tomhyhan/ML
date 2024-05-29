import torch

class DeepConvNet:
    """
        Deep convolution neural network. Each layer has following architecture except the final layer:
        
        Conv - BatchNorm - Relu - Maxpool?
        
        final Layer is FullyConnectedLayer to apply softmax.
    """
    
    def __init__(self, input_dim, filters, n_classes, reg, device="cpu", dtype=torch.float):
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
        C, H, W = input_dim
        self.num_layers = len(filters) + 1
        
        for l in range(1, self.num_layers):
            filter, is_maxpool = filters[l-1]
            self.params[f"W{l}"] = None
            self.params[f"B{l}"] = None
        
        
        
        pass