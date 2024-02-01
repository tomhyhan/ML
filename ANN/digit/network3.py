import random
import numpy as np
import mnist_loader
from utils import time_checker

class Network:

    def __init__(self, sizes):
        """
        init network will following properties:
        - start with vector of random weights where row is the size of current layer and column is the size of the previous layer
        - start with vector of random biases where row is the size of current layer and column is a single -(Threshold)
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(b,1) for b in sizes[1:]]
        self.weights = [np.random.randn(wj, wk) for wk, wj in zip(sizes[:-1], sizes[1:])]

    def backprop(self, images, validations):
        """
        compute gradient descent for each neurons in the network
        plan
        1. feedforward: compute z and activation for each layer
        2. output error: compute delta for the last layer
        3. backpropagate the error: compute the previous layer's delta using the last layer delta
        4. output: compute the gradient of the cost function 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.biases]

        zs = []
        activations = [images]
        activation = images
        for w, b in zip(self.weights, self.biases):
            z = np.transpose(np.dot(w, activation), (1,0,2)) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (activations[-1] - validations) * sigmoid_prime(zs[-1])
        nabla_b[-1] = np.sum(delta, axis=0)
        nabla_w[-1] = np.sum(np.matmul(delta, np.transpose(activations[-2], (0,2,1))), axis=0)

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.transpose(np.dot(self.weights[-layer+1].transpose(), delta), (1,0,2)) * sp
            nabla_b[-layer] = np.sum(delta, axis=0)
            nabla_w[-layer] = np.sum(np.matmul(delta, np.transpose(activations[-layer-1], (0,2,1))), axis=0)
            
        return nabla_b, nabla_w
        
    def update_mini_batch(self, mini_batchs, learning_rate):
        """
        use mini batch to compute gradient descent for weight and biases
        1. init nabla biases and nsbla weights
        2. iterate through each mini_batch and apply the back propagation algorithm. This will give us the sum of all the delta nabla weight and biases
        3. Use the result from backpropagation to perform gradient descent algorithm to adjust the weight and biases from each layer.
        """
        images = [image for image, _ in mini_batchs]
        validations = [valids for _, valids in mini_batchs]

        nabla_b, nabla_w = self.backprop(images, validations)
        
        self.biases = [b - (learning_rate / len(mini_batchs)) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (learning_rate) / len(mini_batchs) * nw for w, nw in zip(self.weights, nabla_w)]
        
    @time_checker
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """
        stochastic gradient descent
        input:
        training data: (x,y) x: image y: desired output
        epochs: iteration over entire training data
        1. prepare training data and test data
        2. iterate through the epochs
        3. divide data in to random batches
        4. use mini_batch to update the weight and biases
        5. if there is test data, evaluate the test data against current trained result
        """
        training_data = list(training_data)
        training_data_len = len(training_data)

        if test_data:
            test_data = list(test_data)
            test_data_len = len(test_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batchs = [training_data[k:k+mini_batch_size] for k in range(0, training_data_len, mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, learning_rate)
            
            if test_data:
                print(f"Epoch {epoch} : {self.evaluate(test_data)} /{test_data_len}")
            else:
                print(f"Epoch {epoch} complete!")

    def feedforward(self, activations):
        for w, b in zip(self.weights, self.biases):
            activations = sigmoid(np.dot(w, activations) + b)
        return activations

    def evaluate(self, test_data):
        """
        check if the network correctly outputs the matching result. 
        Use sigmoid squash function with weights, activations and bias to predict correct Number from the network 
        """
        test_results = [(np.argmax(self.feedforward(image)), valid) for image, valid in test_data]
        return sum([predict == valid for predict, valid in test_results])
        
    def evaluate_single(self, image):
        return np.argmax(self.feedforward(image))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 15, 10, 10])
net.SGD(training_data, 3, 10, 3.0, test_data=test_data)