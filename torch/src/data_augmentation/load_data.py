import torchvision 
import torch
import numpy as np
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random

def tensor_to_image(tensor: torch.Tensor):
    """
        transform tensor to image for visualization
    """
    t = tensor.mul(255).add_(0.5).clamp(0,255).permute(1,2,0)
    ndarr = t.to(device="cpu", dtype=torch.uint8).numpy()
    return ndarr 

def data_to_tensor(data, n_samples=None, dtype=torch.float32):
    """
        transform image data to pytorch Tensor    
    """
    X = torch.tensor(data.data, dtype=dtype).permute(0,3,1,2).div(255)
    Y =  torch.tensor(data.targets, dtype=torch.int64)
    if n_samples is not None:
        X = X[:n_samples].clone()
        Y = Y[:n_samples].clone()
    return X, Y

def data_preprocess(image_show=False, n_samples=None, validation_ratio=0.2, dtype=torch.float32):
    """
        load CIFAR10 dataset using torchvision.
        
        if image_show is True, show random images form dataset
        
        Compute the mean RGB from train set and us it to zero center the traning, validation and test set.
    """
    cifar10_train = torchvision.datasets.CIFAR10("./", download=True, train=True)
    X_train, Y_train = data_to_tensor(cifar10_train, n_samples=n_samples, dtype=dtype)
    
    cifar10_train = torchvision.datasets.CIFAR10("./", download=True, train=False)
    X_test, y_test = data_to_tensor(cifar10_train, dtype=dtype)

    if image_show:   
        images_per_class = 5
        
        samples = []
        classes = 10

        for c in range(classes):
            (y_class,) = (Y_train == c).nonzero(as_tuple=True)
            plt.text(-4, c*34 + 30, f"{c}", ha="right")
            for _ in range(images_per_class):
                ridx = y_class[random.randrange(y_class.shape[0])].item()
                samples.append(X_train[ridx])

        img = torchvision.utils.make_grid(samples, nrow=images_per_class)
        plt.imshow(tensor_to_image(img))
        plt.axis("off")
        plt.show()
        
    x_mean = X_train.mean(dim=(0,2,3), keepdim=True)
    X_train -= x_mean
    X_test -= x_mean
    
    n_validations = int(X_train.shape[0] * validation_ratio)
    n_train = X_train.shape[0] - n_validations

    print("use torch randpermute to randomize extracting the training and val set")
    x_train = X_train[:n_train]
    y_train = Y_train[:n_train]
    x_valids = X_train[n_train:n_train+n_validations]
    y_valids = Y_train[n_train:n_train+n_validations]

    return x_train, y_train, x_valids, y_valids, X_test, y_test
    
        
        
# if __name__ == "__main__":
#     data_preprocess()
