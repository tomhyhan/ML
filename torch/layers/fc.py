import torch
from grads import compute_numeric_gradient

class FullyConnectedLayer:

    def __init__(self, in_dim, out_dim, device="cpu", dtype=torch.float32, reg=False):
        """
            initialize the weight and biases of FC layer.
            Employ Xavier Regulaztion if needed
            w: (D,M)
            b: (M)
        """
        eps = 10e-6
        xavier = 1 if not reg else in_dim**0.5
        self.w = torch.randn(in_dim, out_dim, device=device, dtype=dtype) / xavier
        self.b = torch.randn(out_dim, device=device, dtype=dtype)
        self.x = None

    def forward(self, x):
        """
            computes forward pass for fully connected layer
            Inputs:
                x: (N,D)
            outputs:
                out: (N,M) 
        """
        self.x = x
        xw = torch.matmul(self.x, self.w)
        z = xw + self.b
        return z
    
    def backward(self, dout):
        """
            compute backpropagation for fc layer.
            Inputs:
                dout: (N,M) -> upstream gradients
            outputs:
                dx: (N,D)
                dw: (D,M)
                db: (M)
        """
        db = dout.sum(dim=0)
        dw = torch.matmul(self.x.T, dout)
        dx = torch.matmul(dout, self.w.T)
        return dx, dw, db

x = torch.randn(2, 5)    
fc = FullyConnectedLayer(5, 3)


z = fc.forward(x)

dout = torch.randn(z.shape)
dx, dw, db = fc.backward(dout)

f = lambda x: fc.forward(x)
grad = compute_numeric_gradient(f, x, dout)
print(dx)
print(grad)