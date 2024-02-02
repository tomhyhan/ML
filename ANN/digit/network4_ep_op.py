import sys
import random
import json
import numpy as np
import mnist_loader

class QuadraticCost:
    
    @staticmethod
    def delta(a, y, z):
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost:

    @staticmethod
    def delta(a, y, z):
        return (a-y)

class Network:
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.cost = cost
        self.biases, self.weights = self.create_weights_biases(sizes)
    
    def create_weights_biases(self, sizes):
        return [np.random.randn(b,1) for b in sizes[1:]], [np.random.randn(wj,wk) for wk, wj in zip(sizes[:-1], sizes[1:])]         

    def backprop(self, images, valids):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        zs = []
        activations = [images]
        
        activation = images
        for w, b in zip(self.weights, self.biases):
            z = np.transpose(np.dot(w, activation), (1,0,2)) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost.delta(activations[-1], valids, zs[-1])

        nabla_b[-1] = np.sum(delta, axis=0)
        nabla_w[-1] = np.sum(np.matmul(delta, np.transpose(activations[-2], (0,2,1))), axis=0)

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.transpose(np.dot(self.weights[-layer+1].transpose(), delta), (1,0,2)) * sp

            nabla_b[-layer] = np.sum(delta, axis=0)
            nabla_w[-layer] = np.sum(np.matmul(delta, np.transpose(activations[-layer-1], (0,2,1))), axis=0)
        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, learning_rate):
        images = [image for image, _ in mini_batch]
        valids = [valids for _, valids in mini_batch]
        
        nabla_b, nabla_w = self.backprop(images, valids)
        
        self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
    
    def SGD(self, epochs, learning_rate, mini_batch_size, training_data, test_data = None):
        training_data = list(training_data)
        training_data_len = len(training_data)

        if test_data:
            test_data = list(test_data)
            test_data_len = len(test_data)
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batchs = [training_data[k:k+mini_batch_size] for k in range(0,training_data_len, mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, learning_rate)
                
            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} {test_data_len}")
            else:
                print(f"Epoch {epoch} complete!")

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(image)), valid) for image, valid in test_data]
        return sum([predict == result for predict, result in test_results])
    
    def feedforward(self, acivation):
        for w, b in zip(self.weights, self.biases):
            acivation = sigmoid(np.dot(w, acivation) + b)
        return acivation

    def save(self):
        network = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "cost": self.cost.__name__
        }
        with open("network.json", 'w') as f:
            json.dump(network, f)
    
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 15, 10, 10])
net.SGD(3, 0.5, 10, training_data, test_data=test_data)
net.save()