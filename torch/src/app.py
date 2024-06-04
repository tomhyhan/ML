from data_augmentation.load_data import data_preprocess 
from model.DeepConv import DeepConvNet
from solver.solver import Solver
import torch
from src.optimizer.optimizers import adam

if "__main__" == __name__:
    device = "cpu"
    dtype = torch.float32
    n_samples = 100
    
    x_train, y_train, x_valids, y_valids, X_test, y_test = data_preprocess(image_show=False, n_samples=n_samples, validation_ratio=0.2, dtype=dtype)
    
    data = {
        "X_train" : x_train,
        "y_train":  y_train,
        "X_val":    x_valids,
        "y_val":    y_valids,
        "X_test":   X_test,
        "y_test":   y_test
    }
    print(x_train.shape)
    input_dim = x_train[0].shape
    # filters = [[8, True], [16, True]]
    filters = [[8, True], [16, True], [32, True], [64, True]]
    n_classes = 10
    reg = 1e-2
    batchnorm = True
    weight_scale = "kaiming"

    model = DeepConvNet(input_dim, filters, n_classes, reg, batchnorm, weight_scale, device, dtype)

    solver = Solver(model, data, epochs=30, batch_size=10, device=device, 
                    print_every=10,
                    optim_config={
                        'learning_rate': 1e-3,
                    }, 
                    update_rule=adam
                    )
    
    solver.train()
    
    