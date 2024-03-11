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
        self.w = np.random.normal(loc=0, scale=np.sqrt(1/normalize_out), size=self.filter_shape)
        self.b = np.random.normal(loc=0, scale=1, size=(self.filter_shape[0], ))
        self.params = [self.w, self.b]
        print(normalize_out)
        print(self.w.shape)
        print(self.b)

    def set_input(self, input):
        conv = nn.Conv2d(self.filter_shape[1], self.filter_shape[0], (self.filter_shape[2], self.filter_shape[3]))
        image_input = t.randn(self.image_shape)
        conv_out = conv(image_input)
        print(conv_out.shape)

c = ConvPoolLayer(image_shape=(10,1,28,28), filter_shape=(20,1,5,5))
c.set_input(123)
# # Assuming self.b is a tensor
# b = t.randn(10)

# # Rearrange dimensions
# result = b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

# print(result.shape)