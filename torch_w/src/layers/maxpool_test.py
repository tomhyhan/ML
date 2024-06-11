import sys
sys.path.append("../")

import torch
from src.layers.maxpool import Maxpool
from src.utils.test_tools import compute_numeric_gradients, rel_error

def test_gradients():
    device = "cpu"
    dtype = torch.float64
    
    maxpool = Maxpool()
    
    X = torch.randn(10, 3, 16, 16, device=device, dtype=dtype)

    out = maxpool.forward(X)
    
    dout = torch.randn(*out.shape, device=device, dtype=dtype)    
    dx = maxpool.backward(dout)

    fx = lambda x: maxpool.forward(x)
    
    dx_num = compute_numeric_gradients(fx, X, dout)

    dx_error = rel_error(dx, dx_num)
    
    print("dx error: ", dx_error)
    
def test_std_mean():
    pass

if __name__ == "__main__":
    test_gradients()
    
    