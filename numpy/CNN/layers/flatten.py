from base import Layer
import numpy as np

class FlattenLayer(Layer):
    def __init__(self):
        self.shape = None
        
    def forward_pass(self, a_prev, training):
        self.shape = a_prev.shape
        return np.ravel(a_prev).reshape(self.shape[0], -1)
    
    def backward_pass(self, da_curr):
        return da_curr.reshape(self.shape)