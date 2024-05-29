import torch
from batch_norm import BatchNorm
from grads import compute_numeric_gradient, rel_error

def test_mean_std():
    device = "cpu"
    dtype = torch.float64
    
    N, C, H, W = 5,3,14,14
    x = torch.randn(N, C, H, W, device=device, dtype=dtype)
    gamma = torch.ones(C, device=device, dtype=dtype)
    beta = torch.zeros(C, device=device, dtype=dtype)
    batchnormx , _ = BatchNorm.forward(x, gamma, beta)
    
    print(batchnormx.shape)
    print(batchnormx.mean(dim=(0,2,3)))
    print(batchnormx.std(dim=(0,2,3)))
    
    print()
    gamma = torch.tensor([1,2,3], device=device, dtype=dtype)
    beta = torch.tensor([11,12,13], device=device, dtype=dtype)
    batchnormx , _ = BatchNorm.forward(x, gamma, beta)
    
    print(batchnormx.shape)
    print("mean: ", batchnormx.mean(dim=(0,2,3)))
    print("std: ", batchnormx.std(dim=(0,2,3)))
    
def test_test_time():
    """
        test if Batchnorm layer uses running mean and variance in test mode
    """
    device = "cpu"
    dtype = torch.float64
    
    N, C, H, W = 5,3,14,14
    gamma = torch.ones(C, device=device, dtype=dtype)
    beta = torch.zeros(C, device=device, dtype=dtype)
    bn_param = {}
    
    for _ in range(50):
        x = torch.randn(N, C, H, W, device=device, dtype=dtype)
        BatchNorm.forward(x, gamma, beta, bn_param)
    
    x = torch.randn(N, C, H, W, device=device, dtype=dtype)
    batchnormx, _ = BatchNorm.forward(x, gamma, beta, bn_param, mode="test")
    
    print(batchnormx.shape)
    print("mean: ", batchnormx.mean(dim=(0,2,3)))
    print("std: ", batchnormx.std(dim=(0,2,3)))


def test_gradients():
    device = "cpu"
    dtype = torch.float64
    
    N, C, H, W = 5,3,14,14
    x = torch.randn(N, C, H, W, device=device, dtype=dtype)
    gamma = torch.ones(C, device=device, dtype=dtype)
    beta = torch.zeros(C, device=device, dtype=dtype)
    dout = torch.randn(N, C, H, W, device=device, dtype=dtype)
    
    _ , cache = BatchNorm.forward(x, gamma, beta)
    dx, dgamma, dbeta = BatchNorm.backward(dout, cache)

    fx = lambda X: BatchNorm.forward(X, gamma, beta)[0]
    fg = lambda G: BatchNorm.forward(x, G, beta)[0]
    fb = lambda B: BatchNorm.forward(x, gamma, B)[0]
    
    dx_num = compute_numeric_gradient(fx, x, dout)
    dg_num = compute_numeric_gradient(fg, gamma, dout)
    db_num = compute_numeric_gradient(fb, beta, dout)
    
    dx_error = rel_error(dx, dx_num)
    dg_error = rel_error(dgamma, dg_num)
    db_error = rel_error(dbeta, db_num)
    
    print("dx difference in error: ", dx_error)
    print("dx difference in error: ", dg_error)
    print("dx difference in error: ", db_error)

if __name__ == "__main__":
    test_mean_std()
    print("-------------------------")
    test_test_time()
    print("-------------------------")
    test_gradients()

