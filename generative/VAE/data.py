import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

def load_cifar(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  
    ])
    train_data = datasets.CIFAR10(root="./", train=True, transform=transform, download=True)
    test_data = datasets.CIFAR10(root="./", train=False, transform=transform, download=True)
    
    data_size = 50000
    train_size = int(data_size * 0.9)
    val_size = int(data_size * 0.1)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(range(train_size)))
    val_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(range(train_size, data_size)))

    test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=True)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    load_cifar()

