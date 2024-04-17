import numpy as np

class ConvLayer2D:
    def __init__(self, w, b, padding, stride):
        self.w = w
        self.b = b
        self.padding = padding
        self.stride = stride
        self.dw, self.db = None, None
        self.a_prev = None
    
    @classmethod
    def initialize(cls, filters, kernel_shape, padding="valid", stride=1):
        # filter height, width, channel, n_filter
        # 3 3 1 32  
        w = np.random.randn(*kernel_shape, filters) * 0.1
        b = np.random.randn(filters) * 0.1
        return cls(w=w, b=b, padding=padding, stride=stride)

    @property
    def weights(self):
        return self.w, self.b
    
    @property
    def gradients(self):
        if self.dw is None or self.db is None:
            return None
        return self.dw, self.db
    
    def forward_pass(self, a_prev, is_training):
        self.a_prev = np.array(a_prev, copy=True)
        output_shape = self.calculate_output_dims(input_dims=a_prev.shape)
        n, h_in, w_in, _ = a_prev.shape
        _, h_out, w_out, _ = output_shape
        h_f, w_f, c_f, n_f = self.w.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(array=a_prev, pad=pad)
        
        
    def pad(self, array, pad):
        return np.pad(
            array=array,
            pad_width=((0,0), (pad[0], pad[0]), (pad[1], pad[1]), (0,0)),
            mode="edge"
        )
    
    def caculate_output_dims(self, input_dims):
        n, h_in, w_in, _ = input_dims
        h_f, w_f, _, n_f = self.w
        if self.padding == 'same':
            return n, h_in, w_in, n_f
        elif self.padding == "valid":
            h_out = (h_in - h_f) // self.stride + 1
            w_out = (w_in - w_f) // self.stride + 1
            return n, h_out, w_out, n_f
        else:
            raise ValueError("Unsupportted padding value")
        pass
    
    def calculate_pad_dims(self):
        if self.padding == "same":
            h_f, w_f, _, _ = self.w.shape
            return (h_f - 1) // 2, (w_f - 1) // 2
        elif self.padding == "valid":
            return 0,0
        else:
            raise ValueError("Unsupported padding value")

x = np.expand_dims(np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]), axis=3)

y1 = np.array([[[[1]], [[2]],[[3]]],[[[4]],[[5]],[[6]]],[[[7]],[[8]],[[9]]]])

y2 = np.array([[[[1,1]], [[2,2]],[[3,3]]],[[[4,4]],[[5,5]],[[6,6]]],[[[7,7]],[[8,8]],[[9,9]]]])

y3 = np.array([[[[1,1],[1,1]], [[2,2],[2,2]],[[3,3],[3,3]]],[[[4,4],[4,4]],[[5,5],[5,5]],[[6,6],[6,6]]],[[[7,7],[7,7]],[[8,8],[8,8]],[[9,9],[9,9]]]])

print(y1.shape, x.shape)
# print(x[:,0:3,0:3,:] * y1)
print(x[:,0:3,0:3,:, np.newaxis] * y1[np.newaxis,:,:,:])

# print(np.expand_dims(x, axis=3).shape)
# print(x)
# print(np.expand_dims(x, axis=3))
# print(y)
# print(x * y)
# print(x[:,:, np.newaxis] * y[np.newaxis, : ])
# print(x[:,:, np.newaxis])
# print(y[np.newaxis, : ])
# print(y[np.newaxis, :, : ])
# print(np.pad(x, pad_width=((1,1),(1,1)), mode="edge"))