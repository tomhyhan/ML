# import network
# import network_ep

import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# # net = network.Network([784, 15, 10, 10])
# net = network_ep.Network([784, 15, 10, 10])
# net.SGD(training_data, 3, 10, 3.0, evaluation_data=test_data, monitor_evaluation_cost=True)

import cnn
from cnn import Network
from cnn import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

training_data, validation_data, test_data = cnn.load_data_shared()
mini_batch_size = 10
net = Network([
        FullyConnectedLayer(n_in=784, n_out=5),
        SoftmaxLayer(n_in=5, n_out=10)], mini_batch_size)
net.SGD(training_data[:1000], 2, mini_batch_size, 0.1, 
            validation_data, test_data)