import torch
import numpy as np
from data_augmentation.load_data import data_preprocess 
from model.DeepConv import DeepConvNet
from solver.solver import Solver
from src.optimizer.optimizers import adam
import matplotlib.pyplot as plt
from viz.viz import viz_loss_history, viz_training_and_val

if "__main__" == __name__:
    device = "cpu"
    dtype = torch.float32
    n_samples = 50000
    
    x_train, y_train, x_valids, y_valids, X_test, y_test = data_preprocess(image_show=False, n_samples=n_samples, validation_ratio=0.2, dtype=dtype)
    
    small_samples = 50000
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
    filters = [[8, True], [16, True], [32, True], [64, True]]
    n_classes = 10
    reg = 1e-2
    batchnorm = True
    weight_scale = "kaiming"

    model = DeepConvNet(input_dim, filters, n_classes, reg, batchnorm, weight_scale, device, dtype)
    
    solver = Solver(model, data, epochs=2, batch_size=100, device=device, 
                    print_every=1000,
                    optim_config={
                        'learning_rate': 1e-3,
                    }, 
                    update_rule=adam
                    )
    
    solver.train()
    
    # file_path = "./deepconv.pth"
    # model.save(file_path)
    
    # model.load(file_path, torch.float32, "cpu")
    
    # solver = Solver(model, data, epochs=5, batch_size=10, device=device, 
    #                 print_every=1000,
    #                 optim_config={
    #                     'learning_rate': 1e-3,
    #                 }, 
    #                 update_rule=adam
    #                 )
    # solver.train()
    # viz_loss_history(solver.loss_history)
    # viz_training_and_val(solver.training_acc_history, solver.val_acc_history)


    
    
# overfit small with small training data
# from data_augmentation.load_data import data_preprocess 
# from model.DeepConv import DeepConvNet
# from solver.solver import Solver
# import torch
# from src.optimizer.optimizers import adam

# if "__main__" == __name__:
#     device = "cpu"
#     dtype = torch.float32
#     n_samples = 5000
    
#     x_train, y_train, x_valids, y_valids, X_test, y_test = data_preprocess(image_show=False, n_samples=n_samples, validation_ratio=0.2, dtype=dtype)
    
#     small_samples = 50
#     data = {
#         "X_train" : x_train[:small_samples],
#         "y_train":  y_train[:small_samples],
#         "X_val":    x_valids,
#         "y_val":    y_valids,
#         "X_test":   X_test,
#         "y_test":   y_test
#     }
    
#     print("train shape:", data["X_train"].shape)
#     print("valids shape:", data["X_val"].shape)
    
#     input_dim = x_train[0].shape
#     # filters = [[8, True], [16, True]]
#     filters = [[8, True], [16, True], [32, True], [64, True]]
#     n_classes = 10
#     reg = 1e-2
#     batchnorm = True
#     weight_scale = "kaiming"

#     model = DeepConvNet(input_dim, filters, n_classes, reg, batchnorm, weight_scale, device, dtype)
    
#     solver = Solver(model, data, epochs=30, batch_size=10, device=device, 
#                     print_every=10,
#                     optim_config={
#                         'learning_rate': 1e-3,
#                     }, 
#                     update_rule=adam
#                     )
    
#     solver.train()
    
    