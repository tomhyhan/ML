from abc import ABC, abstractmethod

class Layer(ABC):
    @property
    def weights(self):
        return None
    
    @property
    def gradients(self):
        return None
    
    @abstractmethod
    def forward_pass(self, a_prev, training):
        pass
    
    @abstractmethod
    def backward_pass(self, da_curr):
        pass
    
    def set_weights(self, w, b):
        pass
    
