import sys
sys.path.append("../")

import torch
import math
from src.solver.solver import Solver
from src.models.ResNet import ResNet
from src.layers.basicblock import BasicBlock
from src.layers.bottleneck import Bottleneck
from src.utils.test_tools import compute_numeric_gradients, rel_error
from src.data.load_data import load_data
from src.optimizer.optimizers import adam

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


def test_basicblock_overfit():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(n_samples=50000)
    
    samples = 500
    data = {
        "X_train": X_train[:samples],
        "y_train": y_train[:samples],
        "X_val": X_val[:samples],
        "y_val": y_val[:samples],
    }
    
    print("train shape:", data["X_train"].shape)
    print("valids shape:", data["X_val"].shape)
    
    device = "cpu"
    dtype = torch.float32
    
    model = ResNet(BasicBlock, [3,4,6,3], device=device ,dtype=dtype)
    
    solver = Solver(data, model, epochs=15, batch_size=10, optim_params={"learning_rate": 1e-3}, device=device, dtype=dtype, print_every=20, update_rule=adam)
    
    solver.train()

def test_bottleneck_overfit():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(n_samples=1000)
    
    samples = 500
    data = {
        "X_train": X_train[:samples],
        "y_train": y_train[:samples],
        "X_val": X_val[:samples],
        "y_val": y_val[:samples],
    }
    
    print("train shape:", data["X_train"].shape)
    print("valids shape:", data["X_val"].shape)
    
    device = "cpu"
    dtype = torch.float32
    
    model = ResNet(Bottleneck, [3,4,6,3], device=device ,dtype=dtype)
    
    solver = Solver(data, model, epochs=15, batch_size=10, optim_params={"learning_rate": 1e-3}, device=device, dtype=dtype, print_every=20, update_rule=adam)
    
    solver.train()
    
def test_bottleneck_resnext_overfit():
    # has problem with weight initialization in Conv layer
    # Fix: Set Conv layer to initialize weight in parallel w.r.t number of groups. Or simply use pytorch's Conv2d to init weights
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(n_samples=1000)
    
    samples = 500
    data = {
        "X_train": X_train[:samples],
        "y_train": y_train[:samples],
        "X_val": X_val[:samples],
        "y_val": y_val[:samples],
    }
    
    print("train shape:", data["X_train"].shape)
    print("valids shape:", data["X_val"].shape)
    
    device = "cpu"
    dtype = torch.float32
    
    model = ResNet(Bottleneck, [3,4,6,3], device=device ,dtype=dtype, base_width=4, groups=32)
    
    solver = Solver(data, model, epochs=15, batch_size=10, optim_params={"learning_rate": 1e-3}, device=device, dtype=dtype, print_every=20, update_rule=adam)
    
    solver.train()
    
    
if __name__ == "__main__":
    # test_params()
    # test_loss()
    # test_bottleneck_overfit()
    # test_basicblock_overfit()
    test_bottleneck_resnext_overfit()