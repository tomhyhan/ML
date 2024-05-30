from relu import ReLU
from grads import compute_numeric_gradient, rel_error
import torch

def test_gradients():
    dtype = torch.float64
    device = "cpu"
    
    x = torch.randn(10, 10, dtype=dtype, device=device)
    dout = torch.randn(10,10, dtype=dtype, device=device)
    
    _, cache = ReLU.forward(x)
    dx = ReLU.backward(dout, cache)
    fx = lambda X: ReLU.forward(X)[0]
    dx_num = compute_numeric_gradient(fx, x, dout)
    
    dx_error = rel_error(dx, dx_num)
    print("dx_error", dx_error)

    assert(dx_error < 1e-7)

if __name__ == "__main__":
    test_gradients()
