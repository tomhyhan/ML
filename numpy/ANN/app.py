import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import random
from utils import ReLU, Sigmoid, cost_function, loss_function, ReLU_prime, Sigmoid_prime

N_TRAIN_SAMPLES = 5000
# N_TRAIN_SAMPLES = 50000
N_TEST_SAMPLE = 2500
N_VALID_SAMPLES = 250
N_CLASSES = 10
IMAGE_SIZE = 28

((train_x, train_y), (test_x, test_y))  = fashion_mnist.load_data()

train_data = [(tx, ty) for tx, ty in zip(train_x[:N_TRAIN_SAMPLES, :, :], train_y[:N_TRAIN_SAMPLES])]

test_data = [(tx, ty) for tx, ty in zip(train_x[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES+N_TEST_SAMPLE, :, :], train_y[:N_TEST_SAMPLE])]

valid_data = [(vx, vy) for vx, vy in zip(test_x[:N_VALID_SAMPLES, :, :],test_y[:N_VALID_SAMPLES])]

NN_ARCHITECTURE = [
    {"input_dim": 784, "output_dim": 30, "activation": "relu"},
    {"input_dim": 30, "output_dim": 10, "activation": "relu"},
]

def init_layers(nn_architecture, seed = None):
    # if seed:
    #     np.random.seed(seed)

    weights = []
    biases = []
    for arch in nn_architecture:
        w = np.random.randn(arch["output_dim"], arch["input_dim"]) / np.sqrt(arch["input_dim"])
        b = np.random.randn(arch["output_dim"], 1)
        weights.append(w)
        biases.append(b)
    return weights, biases

def full_forward_propagation(images, weights, biases, activate):
    zs = []
    activations = []
    activations.append(images)

    a = images
    for w,b in zip(weights, biases):
        z = np.matmul(w, a) + b
        zs.append(z)
        a = activate(z)
        activations.append(a)
    return a, zs, activations

def backprop(predict, targets, zs, activations, weights, biases, nn_architecture):
    nabla_w = [np.zeros(w.shape) for w in weights]
    nabla_b = [np.zeros(b.shape) for b in biases]

    targets = targets.reshape(-1, 1)
    delta = loss_function(predict, targets) * ReLU_prime(zs[-1])
    # delta = loss_function(predict, targets) * Sigmoid_prime(zs[-1])
    
    nabla_w[-1] = np.sum(np.matmul(delta, np.transpose(activations[-2], (0,2,1))) , axis=0) 
    nabla_b[-1] = np.sum(delta , axis=0)
    # nabla_b -1 (10, 1)
    # nabla_w -1 (10, 30)
    # delta: (10, 10, 1)
    # nabla_b -1 (30, 1)
    # nabla_w -1 (30, 784)
    # delta: (10, 30, 1)
    for l in range(2, len(nn_architecture) + 1):
        predict = activations[-l]
        curr_weights = weights[-l+1]
        
        delta = np.matmul(curr_weights.transpose(), delta)

        nabla_w[-l] = np.sum(np.matmul(delta, np.transpose(activations[-l-1], (0,2,1))), axis=0) 
        nabla_b[-l] = np.sum(delta, axis=0) 

    return nabla_b, nabla_w
        
def step(weights, biases, nabla_b, nabla_w, lr, batch_size):
    weights = [(1-lr*(0.05 / batch_size)) * w - (lr / batch_size) * nw for w, nw in zip(weights, nabla_w)]
    biases = [b - (lr / batch_size) * nb for b, nb in zip(biases, nabla_b)]
    return weights, biases

def clac_accuracy(predict, targets_test):
    # print("argmax:", np.argmax(a, axis=1).shape)
    # print(y.shape)
    # print(np.argmax(a, axis=1) == y)
    # print(np.argmax(a[0], axis=1) == y[0])
    # return np.sum(np.argmax(a, axis=1) == y)
    print(predict[0])
    print("argmax:", np.argmax(predict[0]))
    results = [(np.argmax(x), y)
                        for (x, y) in zip(predict, targets_test)]
    # print(results[0])
    return sum([int(a == y) for a, y in results])

def sgd(train_data, test_data, valid_data, nn_architecture, epochs, lr, batch_size=10, activation_fn=ReLU):
    weights, biases = init_layers(nn_architecture)
    len_t = len(train_data)
    # print(weights[0])

    for epoch in range(epochs):
        random.shuffle(train_data)
        mini_batches = [train_data[k: k + batch_size] for k in range(0,N_TRAIN_SAMPLES,batch_size)]
        print(f"Epoch {epoch} has started")
        # print(len(mini_batches))
        for i, mini_batch in enumerate(mini_batches):
            if i % 1000 == 0:
                print(f"{i * batch_size} has been trained" )
            images = np.array([image for image, _ in mini_batch]).reshape(batch_size, 784, 1)
            targets = np.array([target for _, target in mini_batch])
            # print(images.shape, targets.shape)
            # feed forward
            predict, zs, activations = full_forward_propagation(images, weights, biases, activation_fn)

            # backward prop
            nabla_b, nabla_w = backprop(predict, targets, zs, activations, weights, biases, nn_architecture)

            # update
            weights, biases = step(weights, biases, nabla_b, nabla_w, lr, batch_size)

            # print(weights[0])
        # print(weights[0])
        # print(np.array([image for image in valid_data[0]]).shape)
    valid_images = np.array([image for image, _ in valid_data]).reshape(N_VALID_SAMPLES, 784, 1)
    valid_targets = np.array([y for _, y in valid_data])
    predict, _, _ = full_forward_propagation(valid_images, weights, biases, activation_fn)
    # print(targets_test.shape, predict.shape)
    result = clac_accuracy(predict, valid_targets)
    print(f"corrections: {result} / {N_VALID_SAMPLES}")
    print(f"accuracy: {result / N_VALID_SAMPLES}")
    # cost = cost_function(predict, valid_targets) / N_VALID_SAMPLES
    # print(f"cost: {cost}")
sgd(train_data, test_data, valid_data, NN_ARCHITECTURE, 3, 0.3, batch_size=10, activation_fn=ReLU)