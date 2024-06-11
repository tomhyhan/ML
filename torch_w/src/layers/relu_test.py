import sys
sys.path.append("../")

import torch
from src.layers.relu import ReLU
from src.utils.test_tools import compute_numeric_gradients, rel_error

def test_gradients():
    device = "cpu"
    dtype = torch.float64
    
    relu = ReLU(device=device, dtype=dtype)
    
    X = torch.randn(15, 32, device=device, dtype=dtype)
    out = relu.forward(X)
    
    dout = torch.randn(*out.shape, device=device, dtype=dtype)
    
    dx = relu.backward(dout)
    
    fx = lambda x: relu.forward(x)
    dx_num = compute_numeric_gradients(fx, X, dout)
    
    dx_error = rel_error(dx, dx_num)
    print("dx error:", dx_error)

if __name__ == "__main__":
    test_gradients()