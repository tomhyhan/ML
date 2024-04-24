from layers.base import Layer
import numpy as np

class SoftmaxLayer(Layer):
    def __init__(self):
        self.z = None

    def forward_pass(self, a_prev, is_training):
        e = np.exp(a_prev - a_prev.max(axis=1, keepdims=True))
        self.z = e / np.sum(e, axis=1, keepdims=True)
        return self.z 
    
    def backward_pass(self, da_curr):
        return da_curr



