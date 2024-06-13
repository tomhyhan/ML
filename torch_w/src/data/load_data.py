import torch
from torchvision.datasets import CIFAR10

def image_to_tensor(data, dtype, n_samples):
    """
        transforms 
            1. x image data to tensor
            2. y labels to tensor 
    """
    
    X, y = data.data, data.targets

    mask = torch
    
    X = torch.tensor(X, dtype=dtype).permute(0,3,1,2).div(255)
    y = torch.tensor(y, dtype=torch.int64)

    if n_samples is not None:
        N = X.shape[0]
        mask = torch.randperm(N)[:n_samples]
        X = X[mask].clone()
        y = y[mask].clone()
    
    return X, y

def load_data(dtype=torch.float32, validation_ratio=0.2, n_samples=None):
    """
        load cifar10 data set and transform it to torch tensor. The training set is divided further into validation set depending on the validation ratio. When n_samples is not None, returns data size w.r.t. to n_samples.
    """
    
    train_data = CIFAR10("./", train=True, download=True)
    test_data = CIFAR10("./", train=False, download=False)
    
    X_train, y_train = image_to_tensor(train_data, dtype, n_samples)
    X_test, y_test = image_to_tensor(test_data, dtype, n_samples)
    
    # zero center data
    x_mean = X_train.mean(dim=(0,2,3))
    X_train -= x_mean
    X_test -= x_mean
    
    N = X_train.shape[0]
    
    n_val = int(validation_ratio * N)
    n_train = N - n_val
    
    mask = torch.randperm(N)
    train_k = mask[:n_train]
    val_k = mask[n_train:n_train+n_val]
       
    x_train_set = X_train[train_k].clone()
    y_train_set = y_train[train_k].clone()
    x_val = X_train[val_k].clone()
    y_val = y_train[val_k].clone()

    return x_train_set, y_train_set, x_val, y_val, X_test, y_test