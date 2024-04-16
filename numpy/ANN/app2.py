import torch as t
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
from tensorflow.keras.datasets import mnist
import random

((train_x, train_y), (test_x, test_y)) = mnist.load_data()

N_TRAIN_DATA = 5000
N_TEST_DATA = 1000
N_VALID_DATA = 250
N = 28 * 28

eps = 10 ** -8

def convert_one_hot(n):
    hot = np.zeros((10,1))
    hot[n] = 1.0
    return hot

def reshape(x):
    return np.reshape(x, (N,1)) / 255
    
def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

def ReLU(z):
    return np.maximum(z, 0)

def Softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    sm = e / np.sum(e, axis=1, keepdims=True)
    return sm
    
def cross_entropy_loss(a, y):
    return - np.sum(y * np.log(np.clip(a, eps, 1))) / N_TRAIN_DATA

def cross_entropy_backward(a, y):
    return a - y 

train_data = [(reshape(x), convert_one_hot(y)) for x, y in zip(train_x[:N_TRAIN_DATA], train_y[:N_TRAIN_DATA])]

test_data = [(reshape(x), y) for x, y in zip(train_x[N_TRAIN_DATA:N_TRAIN_DATA + N_TEST_DATA], train_y[N_TRAIN_DATA:N_TRAIN_DATA + N_TEST_DATA])]

valid_data = [(reshape(x), y) for x, y in zip(train_x[:N_VALID_DATA], train_y[:N_VALID_DATA])]

LAYERS = [
    {"in": 784, "out": 64, "activation": Sigmoid},
    {"in": 64, "out": 10, "activation": Softmax},
]

def init_weights(layers):
    params = {}

    for i, layer in enumerate(layers):
        layer_i = i + 1
        params[f"w{layer_i}"] = np.zeros((layer["out"], layer["in"]))
        params[f"b{layer_i}"] = np.zeros((layer["out"], 1))
        params[f"activation{layer_i}"] = layer["activation"]
        
    return params


def forward_pass(images, layers, params):
    activations = [images]
    zs = []

    a = images
    for l in range(len(layers)):
        l = l + 1
        w, b, activation_fn = params[f"w{l}"], params[f"b{l}"], params[f"activation{l}"]
        z = np.matmul(w, a) + b 
        zs.append(z)
        a = activation_fn(z)
        activations.append(a)
    return a, zs, activations

def backprop(images, targets, params, layers, a, zs, activations):
    nabla_w = [np.zeros(params[f"w{i+1}"].shape) for i in range(len(layers))]
    nabla_b = [np.zeros(params[f"b{i+1}"].shape) for i in range(len(layers))]

    delta = cross_entropy_backward(a, targets)

    nabla_w[-1] = np.sum(np.matmul(delta, np.transpose(activations[-2], (0, 2, 1))), axis=0)
    nabla_b[-1] = np.sum(delta, axis=0)
    
    for l_prev, layer in reversed(list(enumerate(layers[:-1]))):    
        l_curr = l_prev + 1
        w = params[f"w{l_curr+1}"]
        activation_prev = activations[l_prev]
        
        delta = np.matmul(w.transpose(), delta)

        nabla_w[l_prev] = np.sum(np.matmul(delta, np.transpose(activation_prev, (0, 2, 1))), axis=0)
        nabla_b[l_prev] = np.sum(delta, axis=0) 
        
    return nabla_w, nabla_b

def step(nabla_w, nabla_b, params, layers, lr, lmbda, batch_size):
    for i in range(1, len(layers)+1):
        params[f"w{i}"] -= lr / batch_size * nabla_w[i-1]
        params[f"b{i}"] -= lr / batch_size * nabla_b[i-1]

def sgd(train_data, test_data, valid_data, layers, lr=0.05, lmbda=0.5, epochs=3, batch_size=10):
    params = init_weights(LAYERS)    

    for epoch in range(epochs):
        random.shuffle(train_data)
        mini_batches = [train_data[i*batch_size:i*batch_size+batch_size] for i in range(N_TRAIN_DATA//batch_size)]

        print(f"Epoch starting {epoch}...")

        for mini_batch in mini_batches:
            images = np.array([x for x, _ in mini_batch])
            targets = np.array([y for _, y in mini_batch])

            a, zs, activations = forward_pass(images, layers, params)

            nabla_w, nabla_b = backprop(images, targets, params, layers, a, zs, activations)
        
            step(nabla_w, nabla_b, params, layers, lr, lmbda, batch_size)
            
        images = np.array([x for x, _ in valid_data])
        targets = np.array([y for _, y in valid_data])
        a, _, _ = forward_pass(images, layers, params)
        result = np.argmax(a, axis=1)
        print(result.reshape(N_VALID_DATA))
        print(targets.shape)
        print((result == targets))
lr = 0.05
lmbda = 0.5
epochs = 1
batch_size = 10

# sgd(train_data, test_data, valid_data, LAYERS, lr=0.05, lmbda=0.5, epochs=epochs, batch_size=batch_size)

x = np.array([1,2,3,4,5])
y = np.array([1,2,6,4,7])

print(np.sum(x == y))