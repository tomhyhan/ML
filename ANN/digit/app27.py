import cnn_p27
from cnn_p27 import Network
from cnn_p27 import ReLU
from cnn_p27 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = cnn_p27.load_data_shared()
mini_batch_size = 10
net = Network([
        ConvPoolLayer(
            image_shape=(mini_batch_size, 1,28,28), 
            filter_shape=(20,1,5,5),
            poolsize=(2,2),
            activation_fn=ReLU),
        ConvPoolLayer(
            image_shape=(mini_batch_size, 20,12,12), 
            filter_shape=(40,20,5,5),
            poolsize=(2,2),
            activation_fn=ReLU),
        FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 5, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)