from fc import FullyConnectedLayer
from src.test_tools.grads import compute_numeric_gradient, rel_error, grad_check_sparse
import torch


def test_gradients():
    """
        Sanity check for testing gradient for fc layer
        
        The relative error should be less than 1e-7
    """
    device = "cpu"
    dtype = torch.float64

    x = torch.randn(10, 30, dtype=dtype, device=device)
    w = torch.randn(30, 20, dtype=dtype, device=device)
    b = torch.randn(20, dtype=dtype, device=device)
    dout = torch.randn(10, 20, dtype=dtype, device=device)
    
    _, cache = FullyConnectedLayer.forward(x,w,b)
    dx, dw, db = FullyConnectedLayer.backward(dout, cache)
    
    fx = lambda X: FullyConnectedLayer.forward(X,w,b)[0]
    fw = lambda W: FullyConnectedLayer.forward(x,W,b)[0]
    fb = lambda B: FullyConnectedLayer.forward(x,w,B)[0]

    dx_num = compute_numeric_gradient(fx, x, dout)
    dw_num = compute_numeric_gradient(fw, w, dout)
    db_num = compute_numeric_gradient(fb, b, dout)
    
    dx_error = rel_error(dx, dx_num)
    dw_error = rel_error(dw, dw_num)
    db_error = rel_error(db, db_num)
    
    print("dx error:", dx_error)
    print("dw error:", dw_error)
    print("db error:", db_error)
    
    grad_check_sparse(fx, x, dout, dx)
    
    assert(dx_error < 1e-7)
    assert(dw_error < 1e-7)
    assert(db_error < 1e-7)
    
if __name__ == "__main__":
    test_gradients()
    
    
    
    
    