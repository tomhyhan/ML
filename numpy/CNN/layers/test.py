import torch
from .base import Layer

class MaxPoolLayer(Layer):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        
        self.a_prev = None
        self.mask = {}
    
    @staticmethod
    def forward_pass(self, a_prev, is_training):
        self.a_prev = torch.copy(a_prev)
        n, h_in, w_in, c = a_prev.shape
        h_p, w_p = self.pool_size
        h_out = ((h_in - h_p) // self.stride) + 1 
        w_out = ((w_in - w_p) // self.stride) + 1
        output = torch.zeros((n, h_out, w_out, c))
        
        HP = pool_param["pool_height"]
        WP = pool_param["pool_width"]
        stride = pool_param["stride"] 
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_p
                w_start = j * self.stride
                w_end = w_start + w_p
                
                a_prev_slice = a_prev[:, h_start:h_end, w_start:w_end,:]

                self.save_mask(a_prev_slice, (i,j))
                
                max_val = torch.max(a_prev_slice, axis=(1,2))
                
                output[:, :,i,j] = max_val          
        return output
        
    def backward_pass(self, da_curr):
        n, h_in, w_in, c = self.a_prev.shape
        h_p, w_p = self.pool_size
        n, h_out, w_out, _ = da_curr.shape

        output = torch.zeros_like(self.a_prev)
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_p
                w_start  = i * self.stride
                w_end = w_start + w_p
                
                mask = self.mask[(i,j)]

                output[:, h_start:h_end, w_start:w_end, :] += da_curr[:,i:i+1, j:j+1,:] * mask

        return output

    def save_mask(self, x, coord):
        mask = torch.zeros_like(x)
        n, h_x, w_x, c = x.shape
        
        idx = torch.argmax(x.reshape(n, h_x * w_x, c), axis=1)

        mask.reshape(n, h_x * w_x, c)[:,idx,:] = 1
        self.mask[coord] = mask
    
x_shape = torch.tensor((2, 3, 4, 4))
x = torch.linspace(-0.3, 0.4, steps=torch.prod(x_shape), dtype=torch.float64, device='cuda').reshape(*x_shape)
pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

out, _ = MaxPoolLayer.forward(x, pool_param)