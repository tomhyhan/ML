import sys
import random
import json
import numpy as np
import mnist_loader

class QuadraticCost:
    @staticmethod
    def cost_fn(a,y):
        return 0.5 * np.sum((y-a) ** 2)
    
    @staticmethod
    def delta(a, y, z):
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost:
    @staticmethod
    def cost_fn(a,y):
        return -np.sum(np.nan_to_num(y * np.log(a) + (1-y) * np.log(1-a)))

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

    def update_mini_batch(self, mini_batch, learning_rate, lmbda, n):
        images = [image for image, _ in mini_batch]
        valids = [valids for _, valids in mini_batch]
        
        nabla_b, nabla_w = self.backprop(images, valids)
        
        self.weights = [(1- learning_rate * (lmbda/n)) * w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
    
    def SGD(self, epochs, learning_rate, mini_batch_size, training_data, test_data = None, early_stopping_n = 0, monitor_training_cost= False, monitor_training_accuracy=False, monitor_test_cost=False, monitor_test_accuracy=False, lmbda=0):
        training_data = list(training_data)
        training_data_len = len(training_data)

        best_accuracy = 0
        no_accuracy_chage = 0
        current_accuracy = None
        
        if test_data:
            test_data = list(test_data)
            test_data_len = len(test_data)
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batchs = [training_data[k:k+mini_batch_size] for k in range(0,training_data_len, mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, learning_rate, lmbda, training_data_len)

            print(f"Epoch {epoch} complete")
            print("Report: ---------------")
            if monitor_training_cost:
                cost = self.current_cost(training_data, lmbda)
                print(f"Training cost: {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                print(f"Training Data Accuracy: {accuracy} / {training_data_len}")
            if monitor_test_cost:
                cost = self.current_cost(test_data, lmbda, convert=True)
                print(f"Test cost: {cost}")
            if monitor_test_accuracy:
                accuracy = self.accuracy(test_data)
                current_accuracy = accuracy
                print(f"Test Data Accuracy: {accuracy} / {test_data_len}")

            if early_stopping_n > 0:
                if current_accuracy is None:
                    print("please turn on test accuracy mode")
                    return

                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    no_accuracy_chage = 0
                    print(f"Best Accuracy so far is {best_accuracy}")
                else:
                    no_accuracy_chage += 1
                
                if early_stopping_n == no_accuracy_chage:
                    print(f"Early Exit: no accuracy changed in {no_accuracy_chage}")
                    return
        
        
    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(image)), np.argmax(valid)) for image, valid in data]
        else:
            results = [(np.argmax(self.feedforward(image)), valid) for image, valid in data]
        return sum([predict == result for predict, result in results])
    
    def current_cost(self, training_data, lmbda, convert=False):
        cost = 0
        for image, validations in training_data:
            if convert:
                validations = vectorized_result(validations)
            a = self.feedforward(image)
            cost += self.cost.cost_fn(a, validations)
            cost += 0.5*(lmbda/len(training_data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost / len(training_data)
        
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

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 100, 10])
net.SGD(50, 0.5, 10, training_data, test_data=test_data, monitor_training_cost=False, monitor_training_accuracy=False, monitor_test_cost=False, monitor_test_accuracy=True, lmbda=5)
net.save()