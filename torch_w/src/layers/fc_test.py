import sys
sys.path.append("../")

import torch
from src.layers.fc import FullyConnectedLayer
from src.utils.test_tools import compute_numeric_gradients, rel_error

def test_gradients():
    device = "cpu"
    dtype = torch.float64
    
    fc = FullyConnectedLayer(32, 10, relu=False, weight_scale=False, device=device, dtype=dtype)
    
    X = torch.randn(15, 32, device=device, dtype=dtype)
    out_x = fc.forward(X)
    
    dout = torch.randn(*out_x.shape, device=device, dtype=dtype)    
    dx = fc.backward(dout)

    fx = lambda x: fc.forward(x)
    f_inner = lambda _: fc.forward(X)
    
    dx_num = compute_numeric_gradients(fx, X, dout)
    dw_num = compute_numeric_gradients(f_inner, fc.w, dout)
    db_num = compute_numeric_gradients(f_inner, fc.b, dout)

    dx_error = rel_error(dx, dx_num)
    dw_error = rel_error(fc.grads['w'], dw_num)
    db_error = rel_error(fc.grads['b'], db_num)
    
    print(dw_error)
    print(dx_error)
    print(db_error)
    
def test_std_mean():
    pass

if __name__ == "__main__":
    test_gradients()
    
    
#     import sys
# sys.path.append("../")

# import torch
# from src.layers.fc import FullyConnectedLayer
# from src.utils.test_tools import compute_numeric_gradients, rel_error, compute_inner_numeric_gradients

# def test_gradients():
#     device = "cpu"
#     dtype = torch.float64
    
#     fc = FullyConnectedLayer(32, 10, relu=False, weight_scale=False, device=device, dtype=dtype)
    
#     X = torch.randn(15, 32, device=device, dtype=dtype)
#     out_x = fc.forward(X)
    
#     dout = torch.randn(*out_x.shape, device=device, dtype=dtype)    
#     dx = fc.backward(dout)

#     fx = lambda x: fc.forward(x)
#     f_inner = lambda: fc.forward(X)
    
#     dx_num = compute_numeric_gradients(fx, X, dout)
#     dw_num = compute_inner_numeric_gradients(f_inner, fc.w, dout)
#     dw_num2 = compute_numeric_gradients(f_inner, fc.w, dout)
#     db_num = compute_inner_numeric_gradients(f_inner, fc.b, dout)

#     dx_error = rel_error(dx, dx_num)
#     dw_error = rel_error(fc.grads['w'], dw_num)
#     dw_error2 = rel_error(fc.grads['w'], dw_num2)
#     db_error = rel_error(fc.grads['b'], db_num)
    
#     print(dw_error)
#     print(db_error)
#     print(dx_error)

# if __name__ == "__main__":
#     test_gradients()