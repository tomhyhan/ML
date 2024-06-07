import sys
sys.path.append("../")

import torch
from src.layers.averagepool import AveragePool
from src.test_tools.grads import compute_numeric_gradient, rel_error

def test_gradients():
    device = "cpu"
    dtype = torch.float64
    
    x = torch.randn(10, 32, 7, 7, device=device, dtype=dtype)
    
    out, cache = AveragePool.forward(x)
    
    dout = torch.randn(*out.shape, device=device, dtype=dtype)
    dx = AveragePool.backward(dout, cache)
    
    fx = lambda X: AveragePool.forward(X)[0]
    
    dx_num = compute_numeric_gradient(fx, x, dout)
    
    dx_error = rel_error(dx, dx_num)
    print("dx_error: ", dx_error)
    
    
if __name__ == "__main__":
    test_gradients()