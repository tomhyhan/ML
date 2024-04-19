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
        """
            filters: height, weight, channels, n_filters
        """
        w = np.random.randn(*kernel_shape, filters) * 0.1
        b = np.random.randn(filters) * 0.1
        
        return cls(w, b, padding, stride)

    @property
    def weights(self):
        return (self.w, self.b)
    
    @property
    def gradients(self):
        if self.dw is None or self.db is None :
            return None
        return (self.dw, self.db)
    
    def forward_pass(self, a_prev, is_training):
        self.a_prev = np.copy(a_prev)
        output_dims = self.caculate_output_dims(a_prev.shape)
        n, h_out, w_out, _ = output_dims
        h_f, w_f, c, n_f = self.w.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(a_prev, pad)
        output = np.zeros(output_dims)
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_f
                w_start = j * self.stride
                w_end = w_start + w_f
                output[:, i, j, :] = np.sum(
                    a_prev_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] * self.w[np.newaxis, :, :, :],
                    axis=(1,2,3)
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
        _, h_out, w_out, _ = da_curr.shape
        n, h_in, w_in, _ = self.a_prev.shape
        h_f, w_f, _, _ = self.w.shape
        pad = self.calculate_pad_dims()
        a_prev_pad = self.pad(array=self.a_prev, pad=pad)
        output = np.zeros_like(a_prev_pad)

        self.db = da_curr.sum(axis=(0, 1, 2)) / n
        self.dw = np.zeros_like(self.w)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + h_f
                w_start = j * self.stride
                w_end = w_start + w_f
                output[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self.w[np.newaxis, :, :, :, :] *
                    da_curr[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=4
                )
                self._dw += np.sum(
                    a_prev_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] *
                    da_curr[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=0
                )

        self.dw /= n
        return output[:, pad[0]:pad[0]+h_in, pad[1]:pad[1]+w_in, :]
        

    def pad(self, array, pad):
        return np.pad(
            array=array,
            pad_width=((0,0), (pad[0],pad[0]), (pad[1],pad[1]), (0,0)),
            mode="edge"
        )
    
    def caculate_output_dims(self, input_dims):
        n, h_in, w_in, n_f_in = input_dims
        print("shape:", self.w.shape)
        h_f, w_f, n_c, n_f = self.w.shape
        if self.padding == "same":
            return n, h_in, w_in, n_f
        elif self.padding == "valid":
            h_out = (h_in - h_f) // self.stride - 1
            w_out = (w_in - w_f) // self.stride - 1
            return n, h_out, w_out, n_f

    def calculate_pad_dims(self):
        h_f, w_f, n_c, n_f = self.w.shape
        if self.padding == "same":
            h_out = (h_f - 1) // 2 * self.stride
            w_out = (w_f - 1) // 2 * self.stride
            return h_out, w_out
        elif self.padding == "valid":
            return 0, 0 
# x = np.expand_dims(np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]), axis=3)

# y1 = np.array([[[[1]], [[2]],[[3]]],[[[4]],[[5]],[[6]]],[[[7]],[[8]],[[9]]]])

# y2 = np.array([[[[1,1]], [[2,2]],[[3,3]]],[[[4,4]],[[5,5]],[[6,6]]],[[[7,7]],[[8,8]],[[9,9]]]])

# y3 = np.array([[[[1,1],[1,1]], [[2,2],[2,2]],[[3,3],[3,3]]],[[[4,4],[4,4]],[[5,5],[5,5]],[[6,6],[6,6]]],[[[7,7],[7,7]],[[8,8],[8,8]],[[9,9],[9,9]]]])

# print("shape")
# # print(x)
# print(x.sum(axis=(1,2,3)))
# print(x.sum(axis=(0,1,2)))
# print(x.sum(axis=(2, 3)))
# print("--")
# print(x)
# print(x.shape, y1.shape)
# print(x[:,0:3,0:3,:, np.newaxis].shape, y1[np.newaxis,:,:,:].shape)
# print("resulting shape")
# print((x[:,0:3,0:3,:] * y1).shape)
# r = (x[:,0:3,0:3,:, np.newaxis] * y1[np.newaxis,:,:,:])
# print(r)
# print(r.shape)
# print(np.sum(r, axis=(1,2,3)))


# x = np.array([[[[1,2,3],[1,2,3],[1,2,3]]]])
# print(x * x)
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


# Define input data (assuming a 4D tensor representing a batch of images)
input_data = np.random.randn(2, 4, 4, 3)  # Batch size of 2, images of size 28x28 with 3 channels (RGB)

# Create a convolutional layer with specific parameters
filters = 2  # Number of filters (output channels)
kernel_size = (3, 3, 1)  # Size of the filter (kernel)
stride = 1  # Stride of the convolution
padding = "same"  # Padding type ("same" or "valid")

conv_layer = ConvLayer2D.initialize(filters, kernel_size, padding, stride)

# Perform forward pass
output = conv_layer.forward_pass(input_data, is_training=True)  # Set is_training=True for backpropagation

# Print the output shape
print(output.shape)  # Output: (2, 28, 28, 8) - Batch size remains the same, 
                       #         feature maps become 8 due to filters

# Simulate backpropagation (assuming you have error gradients)
# ... (code to calculate error gradients for the output)

# Perform backward pass
da_prev = conv_layer.backward_pass(np.random.randn(2, 4, 4, 3))  # Replace error_gradients with actual values

# Access layer weights and gradients (for monitoring or updates)
weights, gradients = conv_layer.weights, conv_layer.gradients