import numpy as np
import random
import mnist_loader
import time

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
        self.v_weights = [np.zeros((wj,wk)) / np.sqrt(wk) for wk, wj in zip(sizes[:-1], sizes[1:])] 
        self.v_biases = [np.zeros((b,1)) for b in sizes[1:]]

    def default_weight_initializer(self, sizes):
        return [np.random.randn(b,1)  for b in sizes[1:]], [np.random.randn(wj,wk) / np.sqrt(wk) for wk, wj in zip(sizes[:-1], sizes[1:])] 
        
    def large_weight_initializer(self):
        self.weights = [np.random.randn(wj,wk) for wk, wj in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.random.randn(b,1)  for b in self.sizes[1:]] 
    
    
    def SGD(self, epochs, training_data, learning_rate, mini_batch_size, 
    friction=1,
    lmbda=0, 
    test_data=None, 
    early_stopping_n=0,             monitor_evaluation_cost=False,
    monitor_evaluation_accuracy=False,
    monitor_training_cost=False,
    monitor_training_accuracy=False):

        training_data = list(training_data)
        training_data_len = len(training_data)
        
        best_accuracy = 0
        current_accuracy = 0
        no_accuracy_chage = 0
        original_learning_rate = learning_rate

        if test_data:
            test_data = list(test_data)
            test_data_len = len(test_data)
        
        
        for epoch in range(epochs):
            start_time = time.time()
            random.shuffle(training_data)
            mini_batchs = [training_data[k:k+mini_batch_size] for k in range(0,training_data_len,mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, learning_rate, lmbda, training_data_len, friction)
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:0.2f} seconds")

            print("Epoch %s training complete" % epoch)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                print("Accuracy on training data: {} / {}".format(accuracy, training_data_len))
            if monitor_evaluation_cost:
                cost = self.total_cost(test_data, lmbda, convert=True)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(test_data)
                current_accuracy = accuracy
                print("Accuracy on evaluation data: {} / {}".format(accuracy, test_data_len))

            if early_stopping_n > 0:
                if current_accuracy >= best_accuracy:
                    best_accuracy = current_accuracy
                    no_accuracy_chage = 0
                else:
                    print(best_accuracy, current_accuracy)
                    no_accuracy_chage += 1
                
                if early_stopping_n == no_accuracy_chage:
                    learning_rate /= 2
                    print(f"no accuracy changed in {no_accuracy_chage} scaling down the learning rate from {learning_rate * 2} to {learning_rate}")

                if learning_rate == original_learning_rate / 16:
                    print(f"best accuracy {best_accuracy} learning rate: {learning_rate}")                
                    return
            
    def update_mini_batch(self, mini_batchs, learning_rate, lmbda, n, friction):
        root_activations = [a for a, _ in mini_batchs] 
        valids = [v for _, v in mini_batchs] 
        
        nabla_w, nabla_b = self.backprop(root_activations, valids)
        # L2 regularization
        self.v_weights = [friction * wv - learning_rate / len(mini_batchs) * nw for wv, nw in zip(self.v_weights, nabla_w)]
        self.v_biases = [friction * vb - learning_rate / len(mini_batchs) * nb for vb, nb in zip(self.v_biases, nabla_b)]
        self.weights = [(1-learning_rate*(lmbda / n)) * w + vw for w, vw in zip(self.weights, self.v_weights)]
        self.biases = [b + vb for b, vb in zip(self.biases, self.v_biases)]

        # L1 regularization
        # self.weights = [w - (learning_rate*(lmbda / n) * w) - (learning_rate / len(mini_batchs) * nw) for w, nw in zip(self.weights, nabla_w)]
        # self.biases = [b - learning_rate / len(mini_batchs) * nb for b, nb in zip(self.biases, nabla_b)]
        
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

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - to the power of.
        return cost

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()
net = Network([784, 30, 10], cost=CrossEntropyCost)
training_data = list(training_data)
test_data = list(test_data)
net.large_weight_initializer()
net.SGD(100, training_data[:5000], 0.5, 10, friction=0.5, lmbda=0, test_data=test_data[:500], early_stopping_n = 5, monitor_evaluation_accuracy=True, monitor_training_cost=True)

# Epoch 0: 9281 / 10000
# Epoch 1: 9354 / 10000
# Epoch 2: 9324 / 10000
# Epoch 3: 9388 / 10000