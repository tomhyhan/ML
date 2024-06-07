import torch
import numpy as np
from data_augmentation.load_data import data_preprocess 
from model.Resnet import ResNet
from solver.solver2 import Solver2
from src.optimizer.optimizers import adam
import matplotlib.pyplot as plt
from viz.viz import viz_loss_history, viz_training_and_val



if "__main__" == __name__:
    device = "cpu"
    dtype = torch.float32
    n_samples = 5000
    
    x_train, y_train, x_valids, y_valids, X_test, y_test = data_preprocess(image_show=False, n_samples=n_samples, validation_ratio=0.2, dtype=dtype)
    
    small_samples = 50
    data = {
        "X_train" : x_train[:small_samples],
        "y_train":  y_train[:small_samples],
        "X_val":    x_valids,
        "y_val":    y_valids,
        "X_test":   X_test,
        "y_test":   y_test
    }
    
    print("train shape:", data["X_train"].shape)
    print("valids shape:", data["X_val"].shape)
    
    input_dim = x_train[0].shape
    # filters = [[8, True], [16, True]]
    layers = [[2,4],[2,8],[1,16],[1,32]]
    n_classes = 10
    reg = 1e-2

    model = ResNet(input_dim, layers, n_classes, reg, device, dtype)
    
    solver = Solver2(model, data, epochs=15, batch_size=10, device=device, 
                    print_every=20,
                    optim_config={
                        'learning_rate': 1e-3,
                    }, 
                    update_rule=adam
                    )
    
    solver.train()