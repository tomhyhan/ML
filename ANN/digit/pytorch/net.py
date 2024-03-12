import torch as t
from torch import nn 
import numpy as np

x = t.tensor(np.array([1,2,3]))
# print(t.abs(x))

conv = nn.Conv2d(1, 10, 5, stride=2)
print(conv)

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
        
        self.w = t.tensor(np.random.normal(loc=0, scale=np.sqrt(1/normalize_out), size=self.filter_shape)
        )
        self.b = t.tensor(np.random.normal(loc=0, scale=1, size=(self.filter_shape[0], )))

        self.params = [self.w, self.b]

    def set_input(self, input):
        conv = nn.Conv2d(in_channels=self.filter_shape[1], out_channels=self.filter_shape[0], kernel_size=(self.filter_shape[2], self.filter_shape[3]))

        # later input actual image !
        image_input = t.randn(self.image_shape)

        conv_out = conv(image_input)
        pool2d = nn.MaxPool2d((2,2), stride=2)
        pool_out = pool2d(conv_out)
        b_expanded = self.b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.output = self.activation_fn(pool_out + b_expanded)
        self.output_dropout = self.output
        print(self.output_dropout.shape)
    
    
class FullyConnectedLayer:
    def __init__(self, n_in, n_out, activation_fn=t.sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        
        self.w = t.tensor(np.random.normal(loc=0, scale=np.sqrt(1 / n_out), size=(n_in, n_out)))
        self.b = t.tensor(np.random.normal(loc=0, scale=1, size=(n_out, )))
        self.params = [self.w, self.b]
    
    def set_input(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = t.reshape(inpt, (mini_batch_size, self.n_in))
        print(self.inpt.shape, self.w.shape)
        print((0.5 * t.matmul(self.inpt, self.w))[0])
        print((t.matmul(self.inpt, self.w))[0])
        # print(self.activation_fn(np.dot(self.inpt, self.w)) + self.b)
        pass
# first layer
# c = ConvPoolLayer(image_shape=(10,1,28,28), filter_shape=(20,1,5,5))
# ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
#               filter_shape=(40, 20, 5, 5), 
#               poolsize=(2, 2)),
# second layer
c = ConvPoolLayer(image_shape=(10,20,12,12), filter_shape=(40,20,5,5))
c.set_input(10)
print(c.output.shape)
f = FullyConnectedLayer(40 * 4 * 4, 100)
f.set_input(c.output, c.output, 10)
# # Assuming self.b is a tensor
# b = t.randn(10)

# # Rearrange dimensions
# result = b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

# print(result.shape)