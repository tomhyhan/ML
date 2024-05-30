import torch
from maxpool import Maxpool
from grads import compute_numeric_gradient, rel_error

def test_gradients():
    device="cpu"
    dtype=torch.float64
    x = torch.randn(5,3,14,14, device=device, dtype=dtype)
    dout = torch.randn(5,3,7,7, device=device, dtype=dtype)
    
    _, cache = Maxpool.forward(x)
    dx = Maxpool.backward(dout, cache)
    
    fx = lambda X: Maxpool.forward(X)[0]
    dx_num = compute_numeric_gradient(fx, x, dout)
    error = rel_error(dx, dx_num)
    print(error)

if __name__ == "__main__":
    test_gradients()