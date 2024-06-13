import sys
sys.path.append("../")

import torch
from src.layers.basicblock import BasicBlock
from src.utils.test_tools import compute_numeric_gradients, rel_error

def test_size_x():
    device = "cpu"
    dtype = torch.float64
    
    basicblock = BasicBlock(C_in=8, C_out=8, stride=1, device=device, dtype=dtype)
    
    X = torch.randn(10, 8, 16, 16, device=device, dtype=dtype)

    out = basicblock.forward(X)
    print(out.shape)
    
    basicblock = BasicBlock(C_in=8, C_out=16, stride=2, device=device, dtype=dtype)
    
    X = torch.randn(10, 8, 16, 16, device=device, dtype=dtype)

    out = basicblock.forward(X)
    print(out.shape)
    
def test_gradients():
    device = "cpu"
    dtype = torch.float64
    
    basicblock = BasicBlock(C_in=4, C_out=8, stride=2, device=device, dtype=dtype)
    
    X = torch.randn(3, 4, 8, 8, device=device, dtype=dtype)

    out = basicblock.forward(X)
    dout = torch.randn(*out.shape, device=device, dtype=dtype)
    
    dx = basicblock.backward(dout)
    
    fx = lambda x: basicblock.forward(x)
    dx_num = compute_numeric_gradients(fx, X, dout)
    
    dx_error = rel_error(dx, dx_num)
    
    print("dx_error", dx_error)

if __name__ == "__main__":
    test_size_x()
    test_gradients()
