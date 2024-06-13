import sys
sys.path.append("../")

import torch
import math
from src.solver.solver import Solver
from src.models.ResNet import ResNet
from src.layers.basicblock import BasicBlock
from src.utils.test_tools import compute_numeric_gradients, rel_error

def test_params():
    device = "cpu"
    dtype = torch.float32
    
    model = ResNet(BasicBlock, [1,1,1,1], device=device ,dtype=dtype)
    
    data = {
        "X_train": torch.randn(100, 3, 32, 32, device=device, dtype=dtype),
        "y_train": torch.randint(10, (100,), device=device, dtype=torch.int64),
        "X_val": torch.randn(100, 3, 32, 32, device=device, dtype=dtype),
        "y_val": torch.randint(10, (100,), device=device, dtype=torch.int64),
    }
    
    solver = Solver(data, model, epochs=3, batch_size=10, optim_params={"learning_rate": 1e-3}, device=device, dtype=dtype, print_every=20)
    
    solver.train()


if __name__ == "__main__":
    test_params()
