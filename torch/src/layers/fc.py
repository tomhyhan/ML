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
        cache = (x, w, b)
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
        x, w, _ = cache
        N = x.shape[0]
        db = dout.sum(dim=0)
        dw = torch.matmul(x.reshape(N,-1).T, dout)
        dx = torch.matmul(dout, w.T)
        return dx.reshape(x.shape), dw, db
