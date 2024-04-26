import numpy as np
from .base import Layer

class FullyConnectedLayer(Layer):
    def __init__(self, w, b):
        self.w = w
        self.b = b
        
        self.dw, self.db = None, None
        self.a_prev = None

    @classmethod
    def initialize(cls, n_out, n_in):
        w = np.random.randn(n_in, n_out) / np.sqrt(n_out)
        b = np.random.randn(1, n_in)
        return cls(w,b)
    
    @property
    def weights(self):
        return self.w, self.b
    
    @property
    def gradients(self):
        if self.dw is None or self.db is None:
            return None
        return self.dw, self.db

    def set_weights(self, w, b):
        self.w = w
        self.b = b

    def forward_pass(self, a_prev, training):
        # w.s = 3 5 
        # a.s = n 5 
        # b.s = 1 3
        self.a_prev = np.copy(a_prev)
        a = a_prev
        z = np.dot(a, self.w.T) + self.b 
        return z
    
    def backward_pass(self, da_curr):
        # da.s = n 3
        # a.s = n 5
        # w.s = 3 5 
        self.db = np.sum(da_curr, axis=0)
        self.dw = np.dot(da_curr.T, self.a_prev) 
        return np.dot(da_curr, self.w)
    
    