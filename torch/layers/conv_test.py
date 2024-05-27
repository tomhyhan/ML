from conv import Convolution
from grads import compute_numeric_gradient, rel_error
import torch


def test_gradients():
    """
        Sanity check for testing gradient for fc layer
        
        The relative error should be less than 1e-7
    """
    device = "cpu"
    dtype = torch.float64

    x = torch.randn(10, 3, 28, 28, dtype=dtype, device=device)
    filter = torch.randn(8, 3, 3, 3, dtype=dtype, device=device)
    b = torch.randn(8, dtype=dtype, device=device)
    padding = 1
    stride = 1
    dout = torch.randn(10, 8, 28, 28, dtype=dtype, device=device)
    
    _, cache = Convolution.forward(x, filter, b, padding, stride)
    dx, dw, db = Convolution.backward(dout, cache)
    
    fx = lambda X: Convolution.forward(X, filter, b, padding, stride)[0]
    dx_num = compute_numeric_gradient(fx, x, dout)
    
    dx_error = rel_error(dx, dx_num)
    print("dx_error", dx_error)

    
if __name__ == "__main__":
    test_gradients()
    
    
    
    
    