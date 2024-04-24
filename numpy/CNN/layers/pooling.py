import numpy as np
from .base import Layer

class MaxPoolLayer(Layer):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

        self.a_prev = None
        self.cache = {}
    
    def forward_pass(self, a_prev, is_training):
        self.a_prev = np.copy(a_prev)        
        n, h_in, w_in, c = a_prev.shape
        h_p, w_p = self.pool_size
        h_out = 1 + (h_in - h_p) // self.stride
        w_out = 1 + (w_in - w_p) // self.stride
        output = np.zeros((n, h_out, w_out, c))        
    
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_p
                w_start = j * self.stride
                w_end = w_start + w_p
                a_prev_slice = a_prev[:,h_start:h_end, w_start:w_end,:]
                self.save_mask(a_prev_slice, (i,j))
                output[:, i, j, :] = np.max(a_prev_slice, axis=(1,2))
                
        return output

    
    def backward_pass(self, da_curr):
        print("pooling backward", da_curr.shape)
        n, h_out, w_out, n_f = da_curr.shape
        output = np.zeros_like(self.a_prev)
        h_p, w_p = self.pool_size
        print(output.shape)
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_p
                w_start = j * self.stride
                w_end = w_start + w_p

                mask = self.cache[(i,j)]

                output[:,h_start:h_end, w_start:w_end,:] += mask * da_curr[:, i:i+1, j:j+1,:]
                
        return output 


    
    def save_mask(self, x, coord):
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        # [[1 2] [3 4]]
        # [1 2 3 4], idx = 3
        idx = np.argmax(x, axis=1)
        
        n_idx, c_idx = np.indices((n,c))
        mask.reshape(n, h*w, c)[n_idx, idx, c_idx] = 1
        self.cache[coord] = mask
