import random
import numpy as np
import mnist_loader


class Network:

    def __init__(self, sizes):
        """
        init network will following properties:
        - start with vector of random weights where row is the size of current layer and column is the size of the previous layer
        - start with vector of random biases where row is the size of current layer and column is a single -(Threshold)
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(layer_j, layer_k)
                        for layer_k, layer_j in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(layer, 1) for layer in sizes[1:]]

    def backprop(self, root_activations, validations):
        """
        compute gradient descent for each neurons in the network
        plan
        1. feedforward: compute z and activation for each layer
        2. output error: compute delta for the last layer
        3. backpropagate the error: compute the previous layer's delta using the last layer delta
        4. output: compute the gradient of the cost function 
        """
        # 1
        print("shape", root_activations.shape)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = root_activations
        activations = [activation]
        zs = []
        for w, b in zip(self.weights, self.biases):
            print(np.dot(w, activation))
            print(np.dot(w, activation).shape)
            print(np.transpose(np.dot(w, activation), (1,0,2)))
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 2
        delta = (activations[-1] - validations) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            # print(delta)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
        return (nabla_w, nabla_b)  

    def update_mini_batch(self, mini_batch, learning_rate):
        """
        use mini batch to compute gradient descent for weight and biases
        1. init nabla biases and nsbla weights
        2. iterate through each mini_batch and apply the back propagation algorithm. This will give us the sum of all the delta nabla weight and biases
        3. Use the result from backpropagation to perform gradient descent algorithm to adjust the weight and biases from each layer.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        root_activations = np.array([image for image, _ in mini_batch])
        validations = np.array([validation for _, validation in mini_batch])
        # for root_activations, validations in mini_batch:
        nabla_b, nabla_w = self.backprop(
            root_activations, validations)
        # nabla_b = [b + dnb for b, dnb in zip(nabla_b, delta_nabla_b)]
        # nabla_w = [w + dnw for w, dnw in zip(nabla_w, delta_nabla_w)]

        # apply gradient descent nabla C = (w, b)T
        # change in v = - learning rate * nabla
        # print("weight", self.weights)
        self.weights = [w-(learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))* nb for b, nb in zip(self.biases, nabla_b)]

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
        training_data_length = len(training_data)

        if test_data:
            test_data = list(test_data)
            test_data_length = len(test_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, training_data_length, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
                # break
            if test_data:
                print(f"Epoch {epoch} :{self.evaluate(test_data)} / {test_data_length}")
                pass
            else:
                print(f"Epoch: {epoch} complete!")

    def feedforward(self, activations):
        for weights, bias in zip(self.weights, self.biases):
            # print("ff")
            # print(weights.shape, activations.shape)
            activations = sigmoid(np.dot(weights, activations) + bias)
            # print("a", activations.shape)
        return activations

    def evaluate(self, test_data):
        """
        check if the network correctly outputs the matching result. 
        Use sigmoid squash function with weights, activations and bias to predict correct Number from the network 
        """
        testing_result = [(np.argmax(self.feedforward(image)), valids) for (image, valids) in test_data]
        return sum(int(predict == output) for (predict, output) in testing_result)

    def evaluate_single(self, image):

        return np.argmax(self.feedforward(image))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
sample = list(test_data)[0]
net = Network([784, 5, 10])
net.SGD(training_data, 1, 10, 3.0, test_data=test_data)
# print("predict:", net.evaluate_single(sample[0]))
# print("actual:", sample[1])
# print("len:", len(sample[0]))

# w = np.array([[1,2,3],
#               [4,5,6],
#               [7,8,9]])
# # a = np.array([[1],[2],[3]])
# a = np.array([[1,2,3]])

# print(np.dot(a, w))
