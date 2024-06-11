class ReLU:
    """
        ReLU layer
    """
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        

    def forward(self, X):
        """
            Compute forward pass for relu layer
            
            Input:
                X: input data
            Output:
                out: output data. Same size as input data
        """
        self.prev_x = X
        
        mask = X < 0
        relu_x = X.clone()
        relu_x[mask] = 0
        
        return relu_x
    
    def backward(self, dout):
        """
            Computes back propagation for relu layer.
            
            Input:
                dout: upstream gradients
            Output:
                dx: downstream gradients
        """ 
        mask = self.prev_x < 0
        dx = dout.clone()
        dx[mask] = 0  
        return dx
        