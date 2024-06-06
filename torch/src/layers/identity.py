class Identity:
    
    @staticmethod
    def forward(x):
        """
            Forward pass for the identity function. It simply pass the input to output.
        """
        return x
    
    @staticmethod
    def backward(dout):
        """
            back propagation for the identity function. It pass the upstream gradients directly to output gradients.
        """
        
        return dout