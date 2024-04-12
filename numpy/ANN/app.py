import numpy as np
from tensorflow.keras.datasets import mnist
import random
from utils import ReLU, Sigmoid, cost_function, loss_function, ReLU_prime, Sigmoid_prime, simple_loss_function, convert_one_hot

N_TRAIN_SAMPLES = 5000
# N_TRAIN_SAMPLES = 50000
N_TEST_SAMPLE = 2500
N_VALID_SAMPLES = 250
N_CLASSES = 10
IMAGE_SIZE = 28

((train_x, train_y), (test_x, test_y))  = mnist.load_data()

train_data = [(tx / 255, ty) for tx, ty in zip(train_x[:N_TRAIN_SAMPLES, :, :], train_y[:N_TRAIN_SAMPLES])]

test_data = [(tx, ty) for tx, ty in zip(train_x[N_TRAIN_SAMPLES:N_TRAIN_SAMPLES+N_TEST_SAMPLE, :, :], train_y[:N_TEST_SAMPLE])]

valid_data = [(vx / 255, vy) for vx, vy in zip(test_x[:N_VALID_SAMPLES, :, :],test_y[:N_VALID_SAMPLES])]

NN_ARCHITECTURE = [
    {"input_dim": 784, "output_dim": 30, "activation": "relu"},
    {"input_dim": 30, "output_dim": 10, "activation": "relu"},
]

def init_layers(nn_architecture, seed = None):
    # if seed:
    #     np.random.seed(seed)

    weights = []
    biases = []
    activation_fns = []
    for arch in nn_architecture:
        w = np.random.randn(arch["output_dim"], arch["input_dim"]) / np.sqrt(arch["input_dim"])
        b = np.random.randn(arch["output_dim"], 1)
        weights.append(w)
        biases.append(b)
    return weights, biases

def full_forward_propagation(images, weights, biases, activation_fn):
    zs = []
    activations = []
    activations.append(images)

    a = images
    for w,b in zip(weights, biases):
        z = np.matmul(w, a) + b
        zs.append(z)
        a = activation_fn(z)
        activations.append(a)
    return a, zs, activations

def backprop(predict, targets, zs, activations, weights, biases, nn_architecture):
    nabla_w = [np.zeros(w.shape) for w in weights]
    nabla_b = [np.zeros(b.shape) for b in biases]

    targets = np.array(list(map(convert_one_hot, targets)))
    # print(targets.shape)

    delta = simple_loss_function(predict, targets)
    # delta = loss_function(predict, targets) * Sigmoid_prime(zs[-1])
    # delta = loss_function(predict, targets) * ReLU_prime(zs[-1])
    
    nabla_b[-1] = np.sum(delta , axis=0)
    nabla_w[-1] = np.sum(np.matmul(delta, np.transpose(activations[-2], (0,2,1))) , axis=0) 


    for l in range(2, len(nn_architecture) + 1):
        z = zs[-l]
        # sp = Sigmoid_prime(z)

        delta = np.matmul(weights[-l+1].transpose(), delta) 
        # delta = np.matmul(weights[-l+1].transpose(), delta) * sp

        nabla_b[-l] = np.sum(delta, axis=0)
        nabla_w[-l] = np.sum(np.matmul(delta, np.transpose(activations[-l-1], (0,2,1))), axis=0)

    return nabla_b, nabla_w
        
def step(weights, biases, nabla_b, nabla_w, lr, batch_size):
    nweights = [(1-lr*(0.05 / batch_size)) * w - (lr / batch_size) * nw for w, nw in zip(weights, nabla_w)]
    nbiases = [b - (lr / batch_size) * nb for b, nb in zip(biases, nabla_b)]
    return nweights, nbiases

def feedforward(activation, weights, biases):
    for w, b in zip(weights, biases):
        activation = Sigmoid(np.dot(w, activation) + b)
    return activation

def clac_accuracy(data, weights, biases):
    results = [(np.argmax(feedforward(x.reshape(784, 1), weights, biases)), y)
                        for (x, y) in data]
    return sum(int(x == y) for (x, y) in results)

def sgd(train_data, test_data, valid_data, nn_architecture, epochs, lr, activation_fn, batch_size=10):
    weights, biases = init_layers(nn_architecture)

    for epoch in range(epochs):
        random.shuffle(train_data)
        mini_batches = [train_data[k: k + batch_size] for k in range(0,N_TRAIN_SAMPLES,batch_size)]
        print(f"Epoch {epoch} has started")

        for i, mini_batch in enumerate(mini_batches):
            if i % 1000 == 0:
                print(f"{i * batch_size} has been trained" )

            images = np.array([image for image, _ in mini_batch]).reshape(batch_size, 784, 1)
            targets = np.array([target for _, target in mini_batch])

            # feed forward
            predict, zs, activations = full_forward_propagation(images, weights, biases, activation_fn)

            # backward prop
            nabla_b, nabla_w = backprop(predict, targets, zs, activations, weights, biases, nn_architecture)

            # update
            weights, biases = step(weights, biases, nabla_b, nabla_w, lr, batch_size)

        result = clac_accuracy(valid_data, weights, biases)
        print(f"corrections: {result} / {N_VALID_SAMPLES}")
        print(f"accuracy: {result / N_VALID_SAMPLES}")
        # cost = cost_function(predict, valid_targets) / N_VALID_SAMPLES
    # print(f"cost: {cost}")

# apply softmax and ensure dims stay same
    
sgd(train_data , test_data, valid_data, NN_ARCHITECTURE, 10, 0.005, Sigmoid, batch_size=10)

# print(train_data[0][0])