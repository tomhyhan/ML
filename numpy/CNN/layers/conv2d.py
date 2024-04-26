import numpy as np
from .base import Layer

class ConvLayer2D(Layer):
    def __init__(self, w, b, padding, stride):
        self.w = w
        self.b = b
        self.padding = padding
        self.stride = stride
        
        self.dw, self.db = None, None
        self.a_prev = None

    @classmethod
    def initialize(cls, filters, kernel_shape, padding="valid", stride=1):
        """
            filters: height, weight, channels, n_filters
        """
        w = np.random.randn(*kernel_shape, filters) * 0.1 
        b = np.random.randn(filters) * 0.1
        return cls(w, b, padding = padding, stride = stride)

    @property
    def weights(self):
        return (self.w, self.b)
    
    @property
    def gradients(self):
        if self.dw is None or self.db is None:
            return None
        return self.dw, self.db
        
    def set_weights(self, w, b):
        self.w = w
        self.b = b    
    
    def forward_pass(self, a_prev, is_training):
        self.a_prev = np.copy(a_prev)
        n, h_in, w_in, _ = a_prev.shape
        output_shape = self.calculate_output_dims(a_prev.shape)
        n, h_out, w_out, _ = output_shape

        h_f, w_f, c, n_f = self.w.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(a_prev, pad)
        output = np.zeros(output_shape)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_f
                w_start = j * self.stride
                w_end = w_start + w_f


                output[:,i,j,:] += np.sum(
                    a_prev_pad[:,h_start: h_end, w_start:w_end,:,np.newaxis] * self.w[np.newaxis,:,:,:,:], axis=(1,2,3) 
                )
        
        return output + self.b
        
        
    def backward_pass(self, da_curr):
        """
        :param da_curr - 4D tensor with shape (n, h_out, w_out, n_f)
        :output 4D tensor with shape (n, h_in, w_in, c)
        ------------------------------------------------------------------------
        n - number of examples in batch
        w_in - width of input volume
        h_in - width of input volume
        w_out - width of input volume
        h_out - width of input volume
        c - number of channels of the input volume
        n_f - number of filters in filter volume
        """
        # print("da_curr", da_curr.shape)
        n, h_out, w_out, _ = da_curr.shape
        n, h_in, w_in, _ = self.a_prev.shape
        h_f, w_f, _, n_f = self.w.shape

        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(self.a_prev, pad)
        delta = np.zeros_like(a_prev_pad)
        
        self.db = da_curr.sum(axis=(0,1,2)) / n        
        self.dw = np.zeros_like(self.w) 
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_f
                w_start = i * self.stride
                w_end = w_start + w_f
                
                delta[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self.w[np.newaxis,:,:,:,:] * 
                    da_curr[:, i:i+1, j:j+1, np.newaxis, :],
                     axis=4
                )

                self.dw += np.sum(
                    a_prev_pad[:,h_start:h_end,w_start:w_end,:, np.newaxis] * 
                    da_curr[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=0
                )
                
        self.dw /= n
        return delta[:, pad[0]:pad[0]+h_in, pad[1]:pad[1]+h_in, :]

    def pad(self, array, pad):
        return np.pad(
            array=array,
            pad_width=((0,0), (pad[0],pad[0]), (pad[1], pad[1]), (0,0)),
            mode="edge"
        )
    
    def calculate_output_dims(self, input_dims):
        n, h_in, w_in, _ = input_dims
        h_f, w_f, c, n_f = self.w.shape
        
        if self.padding == "same":
            return n, h_in, w_in, n_f
        elif self.padding == "valid":
            h_out = (h_in - h_f) // self.stride + 1
            w_out = (w_in - w_f) // self.stride + 1
            return n, h_out, w_out, n_f
    
    def calculate_pad_dims(self):
        h_f, w_f, _, _ = self.w.shape

        if self.padding == "same":
            h = (h_f - 1) // 2
            w = (w_f - 1) // 2
            return h, w
        elif self.padding == "valid":
            return 0,0
