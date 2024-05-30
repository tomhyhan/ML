import torch

class FullyConnectedLayer:

    @staticmethod
    def forward(x, w, b):
        """
            computes forward pass for fully connected layer
            Inputs:
                x: (N,D)
                w: (D,M)
            outputs:
                out: (N,M) 
        """
        fx = x.clone().reshape(x.shape[0], -1)
        z = torch.matmul(fx, w) + b
        cache = (fx, w, b)
        return z, cache
    
    @staticmethod
    def backward(dout, cache):
        """
            compute backpropagation for fc layer.
            Inputs:
                dout: (N,M) -> upstream gradients
            outputs:
                dx: (N,D)
                dw: (D,M)
                db: (M)
        """
        x, w, b = cache
        db = dout.sum(dim=0)
        dw = torch.matmul(x.T, dout)
        dx = torch.matmul(dout, w.T)
        return dx, dw, db
