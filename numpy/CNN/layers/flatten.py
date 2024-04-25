from .base import Layer
import numpy as np

class FlattenLayer(Layer):
    def __init__(self):
        self.shape = ()
        
    def forward_pass(self, a_prev, training):
        self.shape = a_prev.shape
        # print(np.ravel(a_prev).reshape(self.shape[0], -1).shape)
        return np.ravel(a_prev).reshape(a_prev.shape[0], -1)
    
    def backward_pass(self, da_curr):
        # print("flatten: ", da_curr.shape)
        return da_curr.reshape(self.shape)