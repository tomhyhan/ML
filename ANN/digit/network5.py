import numpy as np
import random
import mnist_loader

class QuadraticCost:
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    def delta(a,y,z):
        return (a - y) * sigmoid_prime(z)
        return 

class CrossEntropyCost:
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    def delta(a,y,z):
        return a - y

class Network:
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.cost = cost
        self.biases, self.weights  = self.default_weight_initializer(sizes)
        
    def default_weight_initializer(self, sizes):
        return [np.random.randn(b,1)  for b in sizes[1:]], [np.random.randn(wj,wk) / np.sqrt(wk) for wk, wj in zip(sizes[:-1], sizes[1:])] 
    
    def SGD(self, epochs, training_data, learning_rate, mini_batch_size, lmbda=0, test_data=None, early_stopping_n=0):

        training_data = list(training_data)
        training_data_len = len(training_data)
        
        best_accuracy = 0
        current_accuracy = 0
        no_accuracy_chage = 0
        
        if test_data:
            test_data = list(test_data)
            test_data_len = len(test_data)
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batchs = [training_data[k:k+mini_batch_size] for k in range(0,training_data_len,mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, learning_rate, lmbda, training_data_len)
        
            if test_data:
                evaluate = self.evaluate(test_data)
                current_accuracy = evaluate / test_data_len 
                
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {test_data_len}")
            else:
                print(f"Epoch {epoch} is complete!")

            if early_stopping_n > 0:
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    no_accuracy_chage = 0
                else:
                    no_accuracy_chage += 1
                
                if early_stopping_n == no_accuracy_chage:
                    print(f"no accuracy changed in {no_accuracy_chage}")
                    return
                    
            
    def update_mini_batch(self, mini_batchs, learning_rate, lmbda, n):
        root_activations = [a for a, _ in mini_batchs] 
        valids = [v for _, v in mini_batchs] 
        
        nabla_w, nabla_b = self.backprop(root_activations, valids)
        # L2 regularization
        self.weights = [(1-learning_rate*(lmbda / n)) * w - learning_rate / len(mini_batchs) * nw for w, nw in zip(self.weights, nabla_w)]
        # L1 regularization
        # self.weights = [w - learning_rate*(lmbda / n) * w - learning_rate / len(mini_batchs) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - learning_rate / len(mini_batchs) * nb for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, root_activations, valids):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        zs = []
        activations = [root_activations]

        activation = root_activations
        for w, b in zip(self.weights, self.biases):
            z = np.transpose(np.dot(w,activation), (1,0,2)) + b
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
        return nabla_w, nabla_b

    def feedforward(self, activation):
        for w, b in zip(self.weights, self.biases):
            activation = sigmoid(np.dot(w, activation) + b)
        return activation

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(image)), predict) for image, predict in test_data]
        return sum([actual == predict for actual, predict in test_results]) 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()
net = Network([784, 20, 10], cost=CrossEntropyCost)
net.SGD(10, training_data, 0.1, 10, lmbda=5, test_data=test_data, early_stopping_n= 2)
# Epoch 0: 9281 / 10000
# Epoch 1: 9354 / 10000
# Epoch 2: 9324 / 10000
# Epoch 3: 9388 / 10000