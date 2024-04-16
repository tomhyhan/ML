import torch as t
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
from tensorflow.keras.datasets import mnist
import random
from utils import ReLU, Sigmoid, cost_function, loss_function, ReLU_prime, Sigmoid_prime, simple_loss_function, convert_one_hot, softmax, eps

N_TRAIN_SAMPLES = 5000
# N_TRAIN_SAMPLES = 50000
N_TEST_SAMPLE = 2500
N_VALID_SAMPLES = 250
N_CLASSES = 10
IMAGE_SIZE = 28

((train_x, train_y), (test_x, test_y)) = mnist.load_data()

train_data = [(tx / 255, ty)
              for tx, ty in zip(train_x[:N_TRAIN_SAMPLES, :, :], train_y[:N_TRAIN_SAMPLES])]

test_data = [(tx, ty) for tx, ty in zip(
    train_x[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES+N_TEST_SAMPLE, :, :], train_y[:N_TEST_SAMPLE])]

valid_data = [(vx / 255, vy)
              for vx, vy in zip(test_x[:N_VALID_SAMPLES, :, :], test_y[:N_VALID_SAMPLES])]

NN_ARCHITECTURE = [
    {"input_dim": 784, "output_dim": 64, "activation_fn": Sigmoid},
    {"input_dim": 64, "output_dim": 10, "activation_fn": softmax},
]


def init_layers(nn_architecture, seed=None):
    # if seed:
    #     np.random.seed(seed)

    weights = []
    biases = []
    activation_fns = []
    for arch in nn_architecture:
        w = np.random.randn(
            arch["output_dim"], arch["input_dim"]) / np.sqrt(arch["input_dim"])
        b = np.random.randn(arch["output_dim"], 1)
        weights.append(w)
        biases.append(b)
        activation_fns.append(arch['activation_fn'])
    return weights, biases, activation_fns


def full_forward_propagation(images, weights, biases, activation_fns):
    zs = []
    activations = []
    activations.append(images)

    a = images
    for w, b, activation_fn in zip(weights, biases, activation_fns):
        z = np.matmul(w, a) + b
        zs.append(z)
        a = activation_fn(z)
        activations.append(a)
    return a, zs, activations


def backprop(predict, targets, zs, activations, weights, biases, nn_architecture):
    nabla_w = [np.zeros(w.shape) for w in weights]
    nabla_b = [np.zeros(b.shape) for b in biases]

    targets = np.array(list(map(convert_one_hot, targets)))

    delta = simple_loss_function(predict, targets)

    nabla_b[-1] = np.sum(delta, axis=0)
    nabla_w[-1] = np.sum(np.matmul(delta,
                         np.transpose(activations[-2], (0, 2, 1))), axis=0)

    for l in range(2, len(nn_architecture) + 1):
        delta = np.matmul(weights[-l+1].transpose(), delta)

        nabla_b[-l] = np.sum(delta, axis=0)
        nabla_w[-l] = np.sum(np.matmul(delta,
                             np.transpose(activations[-l-1], (0, 2, 1))), axis=0)

    return nabla_b, nabla_w


def step(weights, biases, nabla_b, nabla_w, lr, batch_size):
    # (1-lr*(0.05 / batch_size)) *
    nweights = [w - (lr / batch_size) * nw for w, nw in zip(weights, nabla_w)]
    nbiases = [b - (lr / batch_size) * nb for b, nb in zip(biases, nabla_b)]
    return nweights, nbiases


def feedforward(activation, weights, biases, activation_fns):
    for w, b, af in zip(weights, biases, activation_fns):
        activation = af(np.dot(w, activation) + b)
    return activation


def clac_accuracy(data, weights, biases, activation_fns):
    results = [(np.argmax(feedforward(x.reshape(784, 1), weights, biases, activation_fns)), y)
               for (x, y) in data]
    return sum(int(x == y) for (x, y) in results)


def softmax_cross_entropy(data, weights, biases, activation_fns):
    n = len(data)
    cost = 0
    for x, y in data:
        x = feedforward(x.reshape(784, 1), weights, biases, activation_fns)
        y = convert_one_hot(y)
        # print("cost !!")
        # print(x)
        # print(y)
        # print(x * y)
        cost += - np.sum(y * np.log(np.clip(x, eps, 1.))) / N_VALID_SAMPLES
        # print(cost)

    return cost


def sgd(train_data, test_data, valid_data, nn_architecture, epochs, lr, activation_fn, batch_size=10):
    weights, biases, activation_fns = init_layers(nn_architecture)

    for epoch in range(epochs):
        # random.shuffle(train_data)
        mini_batches = [train_data[k: k + batch_size]
                        for k in range(0, N_TRAIN_SAMPLES, batch_size)]
        print(f"Epoch {epoch} has started")

        for i, mini_batch in enumerate(mini_batches):
            if i % 1000 == 0:
                print(f"{i * batch_size} has been trained")

            images = np.array([image for image, _ in mini_batch]).reshape(
                batch_size, 784, 1)
            targets = np.array([target for _, target in mini_batch])

            # feed forward
            predict, zs, activations = full_forward_propagation(
                images, weights, biases, activation_fns)

            # backward prop
            nabla_b, nabla_w = backprop(
                predict, targets, zs, activations, weights, biases, nn_architecture)

            # update
            weights, biases = step(
                weights, biases, nabla_b, nabla_w, lr, batch_size)

        result = clac_accuracy(valid_data, weights, biases, activation_fns)
        print(f"corrections: {result} / {N_VALID_SAMPLES}")
        print(f"accuracy: {result / N_VALID_SAMPLES}")
        cost = softmax_cross_entropy(
            train_data, weights, biases, activation_fns)
        print(f"cost: {cost}")

# apply softmax and ensure dims stay same


sgd(train_data, test_data, valid_data,
    NN_ARCHITECTURE, 2, 0.05, Sigmoid, batch_size=10)


# pytorch implementation


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(784, 64),
            nn.Sigmoid(),
            nn.Linear(64, 10),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        # print("softmax")
        # print(x)
        x = self.softmax(x)
        # print(x)
        return x


batch_size = 10

model = Model()
opt = t.optim.SGD(model.parameters(), lr=0.05)
epochs = 2
loss_fn = nn.CrossEntropyLoss()

for epoch in range(epochs):
    print("Epoch: ", epoch)
    torch_data = DataLoader(train_data, batch_size=batch_size, shuffle=False, )
    torch_valids = DataLoader(
        valid_data, batch_size=N_VALID_SAMPLES, shuffle=False)
    for input, target in torch_data:
        input = input.view(batch_size, -1).float()
        opt.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
    print(loss)
    with t.no_grad():
        for input, target in torch_valids:
            input = input.view(N_VALID_SAMPLES, -1).float()
            output = model(input)
            _, predicted = t.max(output.data, 1)
            total_correct = (predicted == target).sum().item()
            print(f"{total_correct} / {N_VALID_SAMPLES}")
