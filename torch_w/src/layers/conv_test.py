import sys
sys.path.append("../")

import torch
from src.layers.conv import Conv
from src.utils.test_tools import compute_numeric_gradients, rel_error

def test_gradients():
    device = "cpu"
    dtype = torch.float64
    
    conv = Conv(8, 3, k=3, padding=1, device=device, dtype=dtype)
    
    X = torch.randn(5, 3, 16, 16, device=device, dtype=dtype)
    
    out = conv.forward(X)
    dout = torch.randn(*out.shape, device=device, dtype=dtype)
    
    dx = conv.backward(dout)
    fx = lambda x: conv.forward(x)
    f = lambda _: conv.forward(X)
    
    dx_num = compute_numeric_gradients(fx, X, dout)
    dw_num = compute_numeric_gradients(f, conv.w, dout)
    db_num = compute_numeric_gradients(f, conv.b, dout)
    
    dx_error = rel_error(dx, dx_num)
    dw_error = rel_error(conv.grads['w'], dw_num)
    db_error = rel_error(conv.grads['b'], db_num)
    
    print("dx_error", dx_error)
    print("dw_error", dw_error)
    print("db_error", db_error)
    
    pass

if __name__ == "__main__":
    test_gradients()