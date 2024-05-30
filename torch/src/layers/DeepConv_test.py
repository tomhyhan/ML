import torch
from DeepConv import DeepConvNet

def check_initial_loss():
    input_dim = (3,28,28)
    filter = [(8,True),(16,True)]
    n_classes = 10
    reg = 0
    batchnorm = True
    dtype = torch.float64
    device = "cpu"
    
    convnet = DeepConvNet(input_dim, filter, n_classes, reg, batchnorm, dtype=dtype, device=device)
    
    N = 10
    X = torch.randn(N, *input_dim, device=device, dtype=dtype)
    Y = torch.randint(n_classes, (N,), dtype=torch.int64, device=device)
    
    loss , _ = convnet.loss(X,Y)
    print(loss)


if __name__ == "__main__":
    check_initial_loss()