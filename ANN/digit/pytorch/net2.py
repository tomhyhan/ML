import torch as t
import torch.nn as nn
import torchvision
from torch.utils.data import random_split, DataLoader, Subset
from torch.optim import SGD, Adam
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
        # x = self.relu(self.fc1(x))
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
    mnist_train, mnist_test = random_split(mnist_data, [50000,10000])
    return Subset(mnist_train, t.arange(0,5000)), Subset(mnist_test, t.arange(0,100))

def train():
    mnist_train, mnist_test = load_data()
    model = Model()
    # best so far 0.03 0.005 0.5
    # 0.03
    # learning_rate = 0.03
    learning_rate = 0.001
    # 0.1 0.01
    # -> better
    weight_decay = 0.005
    momentum = 0.5
    mini_batch_size = 10
    print(mnist_train[0][0].shape)
    # sgd = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    sgd = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    print(len(mnist_train))
    epches = 5
    for epoch in range(epches):
        mini_batches = DataLoader(mnist_train, mini_batch_size, shuffle=True)
        test_data = DataLoader(mnist_test, len(mnist_test), shuffle=False)

        print(f"starting epoch {epoch}...")
        for i, mini_batch in enumerate(mini_batches):
            if i % 1000 == 0:
                print(f"{i*10} image has been trained")
            images, target = mini_batch
            # print(images.shape)
            # print(target)
            predict = model(images)
            cost = loss_fn(predict, target)

            sgd.zero_grad()
            cost.backward()
            sgd.step()

        with t.no_grad():
            total_loss = 0
            correct = 0
            test_data_l = iter(test_data)
            data, target = next(test_data_l)
            print("test data shape: ", data.shape)
            print("target: ", target)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            pred = output.argmax(1)
            # print("??")
            # print((pred == target))
            # print((pred == target).sum())
            # print((pred == target).sum().item())
            correct += (pred == target).sum().item()
            print("asdf")
            accuracy = correct / len(mnist_test)
        print(f"\nTest Accuracy: {accuracy}")
        # found = 0
        # for data in mnist_test:
        #     image, number = data
        #     predict = model(image.unsqueeze(0))
        #     # print("test")
        #     # print(image.shape)
        #     # print(predict)
        #     # found += t.argmax(predict) == number
        #     found += (t.argmax(predict) == number).float().sum()
        #     # print((t.argmax(predict) == number).float().sum())
        #     # break
        
        # print("accuracy: ", found / len(mnist_test))
        
train()