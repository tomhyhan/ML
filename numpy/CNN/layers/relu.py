import numpy as np
from .base import Layer

class ReLULayer(Layer):
    def __init__(self):
        self.z = None
        
    def forward_pass(self, a_prev, training):
        self.z = np.maximum(a_prev, 0 )
        return self.z
    
    def backward_pass(self, da_curr):
        # try with x > 0 = 1 
        dz = np.array(da_curr, copy=True)
        dz[self.z <= 0] = 0
        return dz
