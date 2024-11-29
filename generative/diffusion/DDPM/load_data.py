from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
import numpy as np

def load_cifar(batch_size=128):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    train_set = datasets.CIFAR10(
        root="./", 
        download=True, 
        train=True, 
        transform=T.Compose(
        T.RandomHorizontalFlip(), 
        T.ToTensor(), 
        T.Normalize(mean=mean, std=std)
    ))
    
    # print(np.mean(np.divide(train_set.data, 255), axis=(0,1,2)))
    # print(np.std(np.divide(train_set.data, 255), axis=(0,1,2)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)    

    return train_loader

if __name__ == "__main__":
    load_cifar()