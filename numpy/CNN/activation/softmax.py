from ..utils import Layer
import numpy as np

class SoftmaxLayer(Layer):
    def forward_pass(self, a_prev):
        # a_prev (n, k)
        e = np.exp(a_prev, axis=1)
        return e / np.sum(e, axis=1)
    
    def backward_pass(self, da_curr):
        return da_curr

def forward_pass(self, a_prev):
    # a_prev (n, k)
    e = np.exp(a_prev, axis=1)
    return e / np.sum(e, axis=1)