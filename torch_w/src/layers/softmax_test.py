import sys
sys.path.append("../")

import torch
import math
from src.layers.softmax import Softmax
from src.utils.test_tools import compute_numeric_gradients, rel_error

def test_loss():
    device = "cpu"
    dtype = torch.float64
    n_classes = 10
    
    X = torch.randn(3023, n_classes, device=device, dtype=dtype) *0.01
    y = torch.randint(n_classes, (X.shape[0],), device=device, dtype=torch.int64)
    softmax = Softmax()
    
    scores = softmax.forward(X)
    loss, _ = softmax.backward(scores, y)

    print(loss.item())
    print(math.log(n_classes))

def test_gradients():
    device = "cpu"
    dtype = torch.float64
    n_classes = 10
    
    X = torch.randn(512, n_classes, device=device, dtype=dtype) * 0.01
    y = torch.randint(n_classes, (X.shape[0],), device=device, dtype=torch.int64)
    softmax = Softmax()
    
    scores = softmax.forward(X)
    _, dx = softmax.backward(scores, y)

    fx = lambda _: softmax.backward(softmax.forward(X), y)[0]
    dx_num = compute_numeric_gradients(fx, X)
    
    dx_error = rel_error(dx, dx_num)
    print("dx error:", dx_error)

if __name__ == "__main__":
    test_loss()
    test_gradients()