from .base import Layer
import numpy as np

class FlattenLayer(Layer):
    def __init__(self):
        self.shape = ()
        
    def forward_pass(self, a_prev, training):
        n = a_prev.shape[0]
        self.shape = a_prev.shape
        return a_prev.reshape(n, -1)
    
    def backward_pass(self, da_curr):
        return da_curr.reshape(self.shape)
