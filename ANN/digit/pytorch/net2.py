import torch as t
import torch.nn as nn
import torchvision
from torch.utils.data import random_split, DataLoader
from torch.optim import SGD

from matplotlib import pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(40 * 4 * 4, 100)
    
        self.fc2 = nn.Linear(100, 10)

        self.softmax = nn.Softmax(dim=1) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.relu(self.pool1(self.conv1(x)))
        x = self.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.softmax(self.fc2(x))
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    # weight init
    # https://stackoverflow.com/questions/77939004/manually-assigning-weights-biases-and-activation-function-in-pytorch
    # what is pytorch default weight init?
    
def load_data():
    mnist_data = torchvision.datasets.MNIST(download=True, root="./", transform=torchvision.transforms.ToTensor())
    print(len(mnist_data))
    mnist_train, mnist_test = random_split(mnist_data, [50000,10000])
    return mnist_train, mnist_test

def train():
    mnist_train, mnist_test = load_data()
    model = Model()
    sgd = SGD(model.parameters(), lr=0.1, weight_decay=0.5, momentum=0.8, )
    # what is damping
    print(model.parameters())
    # epches = 1
    # for epoch in range(epches):
    #     mini_batches = DataLoader(mnist_train, 10, shuffle=True)
    #     for mini_batch in mini_batches:
    #         images, target = mini_batch
    #         output = model(images)
    #         print(output)
    #         print(target)
    #         break

train()