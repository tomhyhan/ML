import numpy as np

class ReLULayer:
    def __init__(self):
        self.z = None
        
    def forward_pass(self, a_prev, is_training):
        self.z = np.maximum(0, a_prev)
        return self.z
    
    def backward_pass(self, da_curr):
        # try with x > 0 = 1 
        dz = np.array(da_curr, copy=True)
        dz[self.z <= 0] = 0
        return dz
