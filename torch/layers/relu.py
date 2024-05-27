class ReLU:
    
    @staticmethod
    def forward(x):
        """
            computes forward pass for relu non-linearity activatio function
            
            Inputs:
                x: tensor 
            Outputs:
                x: tensor
                cache: input x  
        """
        cache = x
        relu_x = x.clone()
        relu_x[x < 0] = 0
        return relu_x, cache
    
    @staticmethod
    def backward(dout, cache):
        """
            computes backward pass for relu layer
            
            Inputs:
                dout: upstream gradients
                cache: input x
            Outpus:
                dout: gradients w.r.t. x
        """
        x = cache
        dout_relu = dout.clone()
        mask = x <= 0
        dout_relu[mask] = 0
        return dout_relu 
        
        
