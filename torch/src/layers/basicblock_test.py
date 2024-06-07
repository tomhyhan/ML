import sys
sys.path.append("../")

import torch
from src.layers.basicblock import BasicBlock
from src.test_tools.grads import rel_error, compute_numeric_gradient

def test_gradients_no_downsample():
    device = "cpu"
    dtype = torch.float64
    C_in = 8
    C_out = 8
    
    X = torch.randn(5, C_in, 8, 8, device=device, dtype=dtype)
    weights = [torch.randn(C_out, C_in, 3, 3, device=device, dtype=dtype), torch.randn(C_out, C_out, 3, 3, device=device, dtype=dtype)]
    biases = [
        torch.zeros(C_out, device=device, dtype=dtype),
        torch.zeros(C_out, device=device, dtype=dtype),
    ]
    gammas = [
        torch.ones(C_out, device=device, dtype=dtype),
        torch.ones(C_out, device=device, dtype=dtype),
    ]
    betas = [
        torch.zeros(C_out, device=device, dtype=dtype),
        torch.zeros(C_out, device=device, dtype=dtype),
    ]
    bn_params = [{},{}]
    dout = torch.randn(5, C_in, 8, 8, device=device, dtype=dtype)
    
    out, cache = BasicBlock.forward(X, weights, biases, stride=1, gammas=gammas, betas=betas, bn_params=bn_params, down_sample=False)
    
    fx = lambda XX: BasicBlock.forward(XX, weights, biases, stride=1,gammas=gammas, betas=betas, bn_params=bn_params)[0]
    
    dx, other_grads = BasicBlock.backward(dout, cache)
    
    dx_num = compute_numeric_gradient(fx, X, dout)
    
    dx_error = rel_error(dx, dx_num)
    print("error:", dx_error)
    
    assert(dx_error < 1e-6)

def test_gradients_downsample():
    device = "cpu"
    dtype = torch.float64
    C_in = 3
    C_out = 16
    
    X = torch.randn(6, C_in, 8, 8, device=device, dtype=dtype)
    weights = [
        torch.randn(C_out, C_in, 3, 3, device=device, dtype=dtype), torch.randn(C_out, C_out, 3, 3, device=device, dtype=dtype),
        torch.randn(C_out, C_in, 1, 1, device=device, dtype=dtype),        
    ]
    biases = [
        torch.zeros(C_out, device=device, dtype=dtype),
        torch.zeros(C_out, device=device, dtype=dtype),
        torch.zeros(C_out, device=device, dtype=dtype),
    ]
    gammas = [
        torch.ones(C_out, device=device, dtype=dtype),
        torch.ones(C_out, device=device, dtype=dtype),
        torch.ones(C_out, device=device, dtype=dtype),
    ]
    betas = [
        torch.zeros(C_out, device=device, dtype=dtype),
        torch.zeros(C_out, device=device, dtype=dtype),
        torch.zeros(C_out, device=device, dtype=dtype),
    ]
    bn_params = [{},{},{}]
    
    dout = torch.randn(6, C_out, 4, 4, device=device, dtype=dtype)
    
    out, cache = BasicBlock.forward(X, weights, biases, stride=2, gammas=gammas, betas=betas, bn_params=bn_params, down_sample=True)
    
    print("forward output shape: ", out.shape)
    
    fx = lambda XX: BasicBlock.forward(XX, weights, biases, stride=2, gammas=gammas, betas=betas, bn_params=bn_params, down_sample=True)[0]
    
    dx, other_grads = BasicBlock.backward(dout, cache)
    
    dx_num = compute_numeric_gradient(fx, X, dout)
    
    dx_error = rel_error(dx, dx_num)
    print("error:", dx_error)
    
    assert(dx_error < 1e-6)
    

if __name__ =="__main__":
    # test_gradients_no_downsample()
    test_gradients_downsample()