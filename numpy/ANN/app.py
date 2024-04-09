import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import random
from utils import ReLU, Sigmoid

N_TRAIN_SAMPLES = 50000
N_TEST_SAMPLE = 2500
N_VALID_SAMPLES = 250
N_CLASSES = 10
IMAGE_SIZE = 28

((train_x, train_y), (test_x, test_y))  = fashion_mnist.load_data()

train_data = [(tr, t) for tr, t in zip(train_x[:N_TRAIN_SAMPLES, :, :], train_y[:N_TRAIN_SAMPLES])]

test_data = [train_x[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES+N_TEST_SAMPLE, :, :], train_y[:N_TEST_SAMPLE]]

valid_data = [test_x[:N_VALID_SAMPLES, :, :],test_y[:N_VALID_SAMPLES]]

NN_ARCHITECTURE = [
    {"input_dim": 784, "output_dim": 30, "activation": "relu"},
    {"input_dim": 30, "output_dim": 10, "activation": "relu"},
]

def init_layers(nn_architecture, seed = None):
    if seed:
        np.random.seed(seed)

    layers = []
    for arch in nn_architecture:
        w = np.random.randn(arch["output_dim"], arch["input_dim"])
        b = np.random.randn(arch["output_dim"], 1)
        layers.append([w, b])
    return layers

def full_forward_propagation(images, params_values, nn_architecture, activate):
    zs = []
    activations = []
    activations.append(images)

    a = images
    for params in params_values:
        w, b = params
        z = np.matmul(w, a) + b
        zs.append(z)
        a = activate(z)
        activations.append(a)
    print(a.shape)
    return a, zs, activations

def sgd(train_data, test_data, valid_data, nn_architecture, epochs, lr, batch_size=10, activate=ReLU):
    params_values = init_layers(nn_architecture)

    len_t = len(train_data)
    # (30, 784) (10, 784, 1)
    # matmul (10, 30, 1)
    
    for epoch in range(epochs):
        random.shuffle(train_data)
        mini_batches = [train_data[i * batch_size: (i + 1) * batch_size] for i in range(len_t // batch_size)]
        print(len(mini_batches))
        for mini_batch in mini_batches:
            images = np.array([image for image, _ in mini_batch]).reshape(batch_size, 784, 1)
            targets = np.array([target for _, target in mini_batch])
            
            # feed forward
            predict, zs, activations = full_forward_propagation(images, params_values, nn_architecture, activate)

            break
            
sgd(train_data, test_data, valid_data, NN_ARCHITECTURE, 1, 0.003, batch_size=10, activate=ReLU)