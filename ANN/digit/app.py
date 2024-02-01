import network
import network_ep

import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = network.Network([784, 15, 10, 10])
net = network_ep.Network([784, 15, 10, 10])
net.SGD(training_data, 3, 10, 3.0, evaluation_data=test_data, monitor_evaluation_accuracy=True)