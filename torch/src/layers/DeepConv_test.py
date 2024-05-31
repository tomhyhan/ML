import torch
from DeepConv import DeepConvNet
import math 
from grads import compute_numeric_gradient, rel_error
import random
def check_initial_loss():
    input_dim = (3,32,32)
    filter = [(8,True),(64,True)]
    n_classes = 10
    reg = 0
    batchnorm = True
    dtype = torch.float64
    device = "cpu"
    
    convnet = DeepConvNet(input_dim, 
                          filter, 
                          n_classes, 
                          reg, 
                          batchnorm,
                        #   weight_scale="kaiming", 
                          dtype=dtype, 
                          device=device)
    
    N = 50
    X = torch.randn(N, *input_dim, device=device, dtype=dtype)
    Y = torch.randint(n_classes, (N,), dtype=torch.int64, device=device)
    
    loss , _ = convnet.loss(X,Y)
    # loss should be around 2.3
    print(loss)
    print(math.log(n_classes))

    convnet.reg = 1
    
    loss , _ = convnet.loss(X,Y)
    
    print(loss)
    print(math.log(n_classes))
    
def test_gradients():
    # input_dim = (3,32,32)
    input_dim = (3,8,8)
    filter = [(8,True),(8,True)]
    n_classes = 10
    batchnorm = True
    dtype = torch.float64
    device = "cpu"
    random.seed(45)
    torch.manual_seed(45)
    N = 10
    X = torch.randn(N, *input_dim, device=device, dtype=dtype)
    Y = torch.randint(n_classes, (N,), dtype=torch.int64, device=device)
    
    # for reg in [0, 3.14]:
    for reg in [0]:
        
        convnet = DeepConvNet(input_dim, filter, n_classes, reg, batchnorm, weight_scale="kaiming", device=device, dtype=dtype)
        
        loss, grads = convnet.loss(X,Y)
        for name in sorted(grads):
            if name != "b2":
                continue
            f = lambda _: convnet.loss(X, Y)[0]
            grad_num = compute_numeric_gradient(f, convnet.params[name], dout=None)
            if name == 'b2':
                print("sum", torch.sum(grad_num).item())
                print("sum:", torch.sum(grads[name]).item())
            # 
            error = rel_error(grads[name], grad_num)
            print(f"relative error for {name} is {error}")
        
    
if __name__ == "__main__":
    # check_initial_loss()
    test_gradients()