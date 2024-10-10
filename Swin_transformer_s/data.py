import torch
from torchvision.datasets import CIFAR10

def _extract_xy(n_data, data):
    X = torch.tensor(data.data, dtype=torch.float32).permute(0,3,1,2).div(255)
    y = torch.tensor(data.targets, dtype=torch.int64)

    N = X.shape[0]
    if n_data:
        indices = torch.randperm(N)[:n_data]
        X = X[indices].clone()
        y = y[indices].clone()

    return X, y

def fetch_cifar10(n_train, n_test):
    train_data = CIFAR10("./", train=True, download=True)
    test_data = CIFAR10("./", train=False)

    train_x, train_y = _extract_xy(n_train, train_data)
    test_x, test_y = _extract_xy(n_test, test_data)

    return train_x, train_y, test_x, test_y
    
def preprocess_cifar10(n_train=None, n_test=None, validation_ratio=0.2):
    train_x, train_y, test_x, test_y = fetch_cifar10(n_train, n_test)
    
    # zero center
    x_train_mean = train_x.mean(dim=(0,2,3), keepdim=True)
    train_x -= x_train_mean
    test_x -= x_train_mean
    
    N = train_x.shape[0]
    
    n_validation_split = int(N * validation_ratio)
    n_train_split = N - n_validation_split
    
    data_dict = {}
    data_dict["X_train"] = train_x[:n_train_split]
    data_dict["y_train"] = train_y[:n_train_split]
    data_dict["X_val"] = train_x[n_train_split:n_train_split + n_validation_split]
    data_dict["y_val"] = train_y[n_train_split:n_train_split + n_validation_split]
    
    data_dict["X_test"] = test_x
    data_dict["y_test"] = test_y
    
    return data_dict