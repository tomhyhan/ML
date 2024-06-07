import sys
sys.path.append("../")

import math
import torch
from src.model.Resnet import ResNet
from src.test_tools.grads import compute_numeric_gradient, rel_error

def test_gradients():
    device = "cpu"
    dtype = torch.float64
    
    input_dim = (3, 16, 16)
    layers = [[2,4],[2,8],[1,16],[1,32]]
    n_classes = 10
    reg = 0
    
    
    X = torch.randn(10, 3, 16, 16, device=device, dtype=dtype)
    y = torch.randint(10, (10,), dtype=torch.int64, device=device)
    
    
    for reg in [0]:
        resnet = ResNet(input_dim, layers, n_classes, reg, device=device, dtype=dtype)
    
        loss, grads = resnet.loss(X, y)
        for name in sorted(grads):
            for i in range(len(grads[name])):
                f = lambda _ : resnet.loss(X,y)[0]
                grad_num = compute_numeric_gradient(f, grads[name][i], dout=None)
                
                error = rel_error(grads[name][i], grad_num)
                print(f"Relative error for {name} is {error}")
                
                break
            break
    
    

if __name__ == "__main__":
    test_gradients()