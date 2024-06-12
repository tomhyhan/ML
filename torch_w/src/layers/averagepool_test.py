import sys
sys.path.append("../")

import torch
from src.layers.averagepool import AveragePool
from src.utils.test_tools import compute_numeric_gradients, rel_error

def test_pool():
    device = "cpu"
    dtype = torch.float64
    
    X = torch.tensor([[[[12,2512],[12,42]]]], device=device, dtype=dtype)
    
    avgpool = AveragePool()
    
    out = avgpool.forward(X)
    dout = torch.ones_like(out)
    
    dx = avgpool.backward(dout)
    print(dx)

if __name__ == "__main__":
    test_pool()