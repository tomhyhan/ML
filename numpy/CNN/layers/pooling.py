import numpy as np
from base import Layer

class MaxPoolLayer(Layer):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = self.stride

        self.a_prev = None
        self.cache = {}
    
    def forward_pass(self, a_prev):
        self.a_prev = np.copy(a_prev)        
        n, h_in, w_in, c = a_prev.shape
        h_p, w_p = self.pool_size
        h_out = 1 + (h_in - h_p) // self.stride
        w_out = 1 + (w_in - w_p) // self.stride
        output = np.zeros((n, h_in, w_in, c))        
    
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

    
    def backward_pass(self):
        pass
    
    def save_mask(self, x, coord):
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        # [[1 2] [3 4]]
        # [1 2 3 4], idx = 3
        idx = np.argmax(x, axis=1)
        
        n_idx, c_idx = np.indices((n,c))
        mask.reshape(n, h*w, c)[n_idx, idx, c_idx] = 1
        print(idx, mask)
        self.cache[coord] = mask

# def save_mask(x, coord):
#     mask = np.zeros_like(x)
#     n, h, w, c = x.shape
#     x = x.reshape(n, h * w, c)
#     # [[1 2] [3 4]]
#     # [1 2 3 4], idx = 3
#     idx = np.argmax(x, axis=1)
    
#     n_idx, c_idx = np.indices((n,c))
#     mask.reshape(n, h*w, c)[n_idx, idx, c_idx] = 1
#     print(mask.reshape(n, h*w, c).shape)
#     print("idx", idx)
#     print("nidx", n_idx)
#     print("cidx", c_idx)
#     print("mask", mask)
#     print("mask", mask.shape)
    
# # print(np.indices((1,1)))
# x = np.array([[[1,2],[3,4]]])
# x = x[:,:,:,np.newaxis]
# # print(x.shape)
# # print(x)

# save_mask(x, (0,0))