import torch as t
import torchvision
import torchvision.transforms as tt
from torch.utils.data import Subset, random_split
from torch.utils.data.dataloader import DataLoader
from torch import nn

def load_data():
    transform = tt.Compose([
        tt.Resize(256),
        tt.CenterCrop(224),
        tt.ToTensor(),
        tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    food101_data =torchvision.datasets.Food101(download=True, root="./", transform=transform)
    print("transform done")
    print(len(food101_data.classes))
    # subset_indices = set()
    # subset_images = []
    # for i in range(len(food101_data)):
    #     if i % 1000 == 0:
    #         print("iter: ", i)
    #         print(subset_indices)
    #     subset_indices.add(food101_data[i][1])
    #     subset_images.append(i)
    #     if len(subset_indices) > 20:
    #         break
    # food101_subset = Subset(food101_data, subset_images)
    # print("len: ", len(food101_subset))
    traning_i = int(len(food101_data) * 0.9)
    test_i = len(food101_data) - traning_i
    f_train, f_test = random_split(food101_data, [traning_i, test_i])
    return f_train, f_test

f_train, f_test = load_data()
print(len(f_train), len(f_test))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.build_net()

    def build_net(self):
        self.dropout = 0.5
        # Sizes
        # image: (batch_size, 3, 224, 224)
        # Conv: (batch_size, 32, 220, 220)
        # Pool: (batch_size, 32, 110, 110)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Sizes
        # image: (batch_size, 32, 110, 110)
        # Conv: (batch_size, 64, 106, 106)
        # Pool: (batch_size, 64, 52, 52)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Sizes
        # image: (batch_size, 64, 52, 52)
        # Conv: (batch_size, 128, 48, 48)
        # Pool: (batch_size, 128, 24, 24)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128* 24 * 24, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.fc2 = nn.Linear(512, 101)

        self.cost_fn = nn.CrossEntropyLoss()
        self.optimizer = t.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        # print("1: ", x.shape)
        out = self.conv1(x)
        # print("2: ", x.shape)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def predict(self, x):
        self.eval()
        return self.forward(x)

    def accuracy(self, x, y):
        predict = self.predict(x)
        result = t.argmax(predict, 1) == y
        return result.sum().item()

    def train_model(self, images, target):
        self.train()
        self.optimizer.zero_grad()
        predict = self.forward(images)
        cost = self.cost_fn(predict, target)
        cost.backward()
        self.optimizer.step()
        return cost
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.build_net()

    def build_net(self):
        self.dropout = 0.5
        # Sizes
        # image: (batch_size, 3, 224, 224)
        # Conv: (batch_size, 32, 220, 220)
        # Pool: (batch_size, 32, 110, 110)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Sizes
        # image: (batch_size, 32, 110, 110)
        # Conv: (batch_size, 64, 106, 106)
        # Pool: (batch_size, 64, 52, 52)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Sizes
        # image: (batch_size, 64, 52, 52)
        # Conv: (batch_size, 128, 48, 48)
        # Pool: (batch_size, 128, 24, 24)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128* 24 * 24, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.fc2 = nn.Linear(512, 101)

        self.cost_fn = nn.CrossEntropyLoss()
        self.optimizer = t.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        # print("1: ", x.shape)
        out = self.conv1(x)
        # print("2: ", x.shape)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def predict(self, x):
        self.eval()
        return self.forward(x)

    def accuracy(self, x, y):
        predict = self.predict(x)
        result = t.argmax(predict, 1) == y
        return result.sum().item()

    def train_model(self, images, target):
        self.train()
        self.optimizer.zero_grad()
        predict = self.forward(images)
        cost = self.cost_fn(predict, target)
        cost.backward()
        self.optimizer.step()
        return cost

# parameters
learning_rate = 0.001
epochs = 5
batch_size = 10
# device = "cuda" if t.cuda.is_available() else "cpu"
device =  "cpu"
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
cnn_model = CNN().to(device)

for epoch in range(epochs):
    mini_batches = DataLoader(f_train, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(f_test, batch_size=len(f_test), shuffle=False)

    print(f"Epoch {epoch} started: ")
    for i, mini_batch in enumerate(mini_batches):
        if i % 1000 == 0:
            print(f"{i*10} images trained complete")
        images, target = mini_batch
        images = images.to(device)
        target = target.to(device)
        print(images.shape)
        print(target)
        cnn_model.train_model(images, target)
        
        
with t.no_grad():
    total_loss = 0
    correct = 0
    for image, target in test_data:
        image = image.to(device)
        target = target.to(device)
        result = cnn_model.accuracy(image, target)
        # total_loss += loss.item()
        # pred = output.argmax(1)
        correct += result
    accuracy = correct / len(test_data)
print(f"\nTest Accuracy: {accuracy}")