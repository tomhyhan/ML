import numpy as np
from .base import Layer

class FullyConnectedLayer(Layer):
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.a_prev = None
        self.dw, self.db = None, None

    @classmethod
    def initialize(cls, n_out, n_in):
        # w - (10, 100)  
        w = np.random.randn(n_in, n_out) * 0.1
        b = np.random.randn(1, n_in) * 0.1
        return cls(w, b)
    
    @property
    def weights(self):
        return self.w, self.b
    
    @property
    def gradients(self):
        if self.dw is None or self. db is None:
            return None
        return self.dw, self.db
    
    def set_weights(self, w, b):
        self.w = w
        self.b = b
        
    def forward_pass(self, a_prev, training):
        self.a_prev = np.copy(a_prev)
        z = np.dot(a_prev, self.w.T) + self.b
        return z 
    
    def backward_pass(self, da_curr):
        # 20 100 , da_curr = 20, 10
        n = self.a_prev.shape[0]
        self.dw = np.dot(da_curr.T, self.a_prev) / n
        self.db = np.sum(da_curr, axis=0, keepdims=True) / n
        return np.dot(da_curr, self.w)
    
    
    