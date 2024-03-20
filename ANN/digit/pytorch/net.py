import torch as t
from torch import nn 
import numpy as np
import gzip
import pickle
import random
from torch.func import grad

class Network(nn.Module):
    def __init__(self, layers, mini_batch_size):
        super(Network, self).__init__()

        self.layers = layers
        self.mini_batch_size = mini_batch_size
        # self.params = [param for layer in self.layers for param in layer.params]
    
    def forward(self, image):
        mini_batch_size = 1
        init_layer = self.layers[0]
        init_layer.set_input(image, image, mini_batch_size)
        
        for l in range(1,len(self.layers)):
            prev_layer, layer = self.layers[l-1], self.layers[l]
            layer.set_input(prev_layer.output, prev_layer.output_dropout, mini_batch_size)

        return layer[-1].output_dropout
        
    
class ConvPoolLayer:
    def __init__(self, image_shape, filter_shape, poolsize=(2,2), activation_fn=t.sigmoid):
        """
        image_shape: tuple containing 4 entries- batch size, num of input filter(image), height, width
        
        filter_shape: tuple containing 4 entries- num of output filters, num of input filters, height, width
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn

        normalize_out = self.filter_shape[0] * np.prod(self.filter_shape[2:]) / np.prod(self.poolsize) 
        
        self.w = t.tensor(np.random.normal(loc=0, scale=np.sqrt(1/normalize_out), size=self.filter_shape), requires_grad=True
        )
        self.b = t.tensor(np.random.normal(loc=0, scale=1, size=(self.filter_shape[0], )), requires_grad=True)

        self.params = [self.w, self.b]

    def set_input(self, inpt, inpt_dropout, mini_batch_size):
        conv = nn.Conv2d(in_channels=self.filter_shape[1], out_channels=self.filter_shape[0], kernel_size=(self.filter_shape[2], self.filter_shape[3]))

        # later input actual image !
        # image_input = t.randn(self.image_shape)
        self.inpt = t.reshape(inpt, self.image_shape)
        print("conv layer input reshape: ", self.inpt.shape)

        conv_out = conv(self.inpt)
        # print(conv_out)
        pool2d = nn.MaxPool2d((2,2), stride=2)
        pool_out = pool2d(conv_out)
        b_expanded = self.b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.output = self.activation_fn(pool_out + b_expanded)
        self.output_dropout = self.output
        print("conv layer output: ", self.output_dropout.shape)
    
class FullyConnectedLayer:
    def __init__(self, n_in, n_out, activation_fn=t.sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        
        self.w = t.tensor(np.random.normal(loc=0, scale=np.sqrt(1 / n_out), size=(n_in, n_out)), requires_grad=True)
        self.b = t.tensor(np.random.normal(loc=0, scale=1, size=(n_out, )), requires_grad=True)
        
        self.params = [self.w, self.b]
    
    def set_input(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = t.reshape(inpt, (mini_batch_size, self.n_in))
        self.output = ((1 - self.p_dropout) * t.matmul(self.inpt, self.w) + self.b)

        # figure out with actual image data 
        self.y_output = t.argmax(self.output, axis=1)

        dropout = nn.Dropout(self.p_dropout)
        self.input_dropout = dropout(t.reshape(inpt_dropout, (mini_batch_size, self.n_in)))
        self.output_dropout = self.activation_fn(t.matmul(self.input_dropout, self.w) + self.b)
    
class SoftmaxLayer:
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        self.w = t.tensor(np.zeros((n_in, n_out)), requires_grad=True)
        self.b = t.tensor(np.zeros((n_out)), requires_grad=True)
        
        self.params = [self.w, self.b]
    
    def set_input(self, inpt, inpt_dropout, mini_bath_size):
        self.inpt = t.reshape(inpt, (mini_bath_size, self.n_in))
        softmax = nn.Softmax(dim=1)
        self.output = softmax((1-self.p_dropout)*t.matmul(self.inpt, self.w) + self.b)
        
        self.y_out = t.argmax(self.output, axis=1)
        # print("self.y_output", self.y_out)
        # print(self.output.shape)
        
        dropout = nn.Dropout(self.p_dropout)
        self.input_dropout = dropout(t.reshape(inpt_dropout, (mini_bath_size, self.n_in)))
        self.output_drop = softmax(t.matmul(self.input_dropout, self.w) + self.b)
        # print((t.matmul(self.input_dropout, self.w) + self.b).shape)
        print("type:", type(self.output_drop))
        
    def cost(self, real_y):
        out = t.log(self.output_drop[t.arange(real_y.shape[0]), real_y])
        # print("out type: ", out)
        # print("out type: ", t.mean(out))
        # return -t.mean(out)
        return -t.mean(out)
        
    def accuracy(self, real_y):
        return t.mean(t.eq(real_y, self.y_out))
    
def load_shared_data():
    mnist_file_path = "../mnist.pkl.gz"
    with gzip.open(mnist_file_path, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    def convert_to_tensor(data):
        image, numbers = data
        image = t.tensor(image)
        numbers = t.tensor(numbers)
        return image, numbers

    training_data = convert_to_tensor(training_data)
    validation_data = convert_to_tensor(validation_data)
    test_data = convert_to_tensor(test_data)
    return training_data, validation_data, test_data
    
training_data, validation_data, test_data = load_shared_data()
# first layer
# c = ConvPoolLayer(image_shape=(10,1,28,28), filter_shape=(20,1,5,5))
# ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
#               filter_shape=(40, 20, 5, 5), 
#               poolsize=(2, 2)),
# second layer

# c = ConvPoolLayer(image_shape=(10,20,12,12), filter_shape=(40,20,5,5))
# c.set_input(10)
# f = FullyConnectedLayer(40 * 4 * 4, 100, p_dropout=0.5)
# f.set_input(c.output, c.output, 10)
# s = SoftmaxLayer(100, 10, p_dropout=0.5)
# s.set_input(f.output, f.output_dropout, 10)

def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda=0.0):
        
        
        
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data
        # print(training_data[0][0].shape)

        num_training_batches = len(training_data[0]) // mini_batch_size 
        num_validation_batches = len(validation_data) // mini_batch_size 
        num_test_batches = len(test_data) // mini_batch_size 
        # print(len(training_x))
        # print(num_training_batches)
        pass
    
        
        # self.layers[-1].cost(self)

        # print(l2_norm_sqaured)
        # print(training_y)
        for epoch in range(epochs):
            for i in range(num_training_batches):
                mini_batch = training_x[i * mini_batch_size: (i+1) * mini_batch_size] 
                mini_batch_y = training_y[i * mini_batch_size: (i+1) * mini_batch_size] 
                
                init_layer = self.layers[0]
                print(len(mini_batch))
                init_layer.set_input(mini_batch, mini_batch, mini_batch_size)
                
                for l in range(1,len(self.layers)):
                    prev_layer, layer = self.layers[l-1], self.layers[l]
                    layer.set_input(prev_layer.output, prev_layer.output_dropout, mini_batch_size)
                print("mini_batch_y: ", mini_batch_y)

                l2_norm_sqaured = sum([(layer.w**2).sum() for layer in self.layers])
                cost = self.layers[-1].cost(mini_batch_y) + 0.5*lmbda*l2_norm_sqaured / num_training_batches
                print("params")
                print(self.params[-1])
                print("backward:", cost.backward())
                print("grad:", cost.grad)
                break
            break
        
mini_batch_size = 10
net = Network([
        ConvPoolLayer(
            image_shape=(mini_batch_size, 1,28,28), 
            filter_shape=(20,1,5,5),
            poolsize=(2,2)),
        FullyConnectedLayer(n_in=20*12*12, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
for name, param in net.named_parameters():
    print(name, param.size())

# net.SGD(test_data=test_data, epochs=10, mini_batch_size=10, eta=0.1, validation_data=validation_data, training_data=training_data, lmbda=0.0)


# # Assuming self.b is a tensor
# b = t.randn(10)

# # Rearrange dimensions
# result = b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

# print(result.shape)

# x = np.array([1,2,3])
# y = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])

# print(np.dot(x,y))