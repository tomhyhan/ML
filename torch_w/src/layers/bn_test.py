import sys
sys.path.append("../")

import torch
from src.layers.bn import BatchNorm
from src.utils.test_tools import compute_numeric_gradients, rel_error

def test_std_mean():
    device = "cpu"
    dtype = torch.float64
    C = 3
    
    bn = BatchNorm(C, device=device, dtype=dtype)

    X = torch.randn(10, C, 16, 16, device=device, dtype=dtype)
    
    out = bn.forward(X)

    print(out.mean(dim=(0,2,3)))
    print(out.var(dim=(0,2,3)))

    bn = BatchNorm(C, device=device, dtype=dtype)

    X = torch.randn(10, C, 16, 16, device=device, dtype=dtype)
    
    print()
    
    bn.w = torch.tensor([1,2,3], device=device, dtype=dtype)
    bn.b = torch.tensor([10,20,30], device=device, dtype=dtype)
    
    out = bn.forward(X)

    print(out.mean(dim=(0,2,3)))
    print(out.var(dim=(0,2,3)))

def test_test_mode():
    device = "cpu"
    dtype = torch.float64
    C = 3
    
    bn = BatchNorm(C, device=device, dtype=dtype)

    for i in range(100):
        X = torch.randn(10, C, 16, 16, device=device, dtype=dtype)
        _ = bn.forward(X)
        
    X = torch.randn(10, C, 16, 16, device=device, dtype=dtype)
    out = bn.forward(X, training=False)

    # mean and std should be slightly larger
    print(out.mean(dim=(0,2,3)))
    print(out.std(dim=(0,2,3)))

def test_gradients():
    device = "cpu"
    dtype = torch.float64
    C = 3
    
    bn = BatchNorm(C, device=device, dtype=dtype)

    X = torch.randn(10, C, 16, 16, device=device, dtype=dtype)
    
    out = bn.forward(X)
    dout = torch.randn(*out.shape, device=device, dtype=dtype)
    
    dx = bn.backward(dout)
    
    fx = lambda x: bn.forward(x)
    f = lambda _: bn.forward(X)
    
    dx_num = compute_numeric_gradients(fx, X, dout)
    dw_num = compute_numeric_gradients(f, bn.w, dout)
    db_num = compute_numeric_gradients(f, bn.b, dout)
    
    dx_error = rel_error(dx, dx_num)
    dw_error = rel_error(bn.grads['w'], dw_num)
    db_error = rel_error(bn.grads['b'], db_num)
    
    print("dx_error", dx_error)
    print("dgamma_error", dw_error)
    print("dbeta_error", db_error)
    
    pass

if __name__ == "__main__":
    test_gradients()
    test_std_mean()
    test_test_mode()