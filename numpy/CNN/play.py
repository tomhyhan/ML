import numpy as np

class Conv2DBackward:
  def __init__(self, a_prev, w, stride):
    self._a_prev = a_prev
    self._w = w
    self._stride = stride
    
  def calculate_pad_dims(self):
    _, h_in, w_in, _ = self._a_prev.shape
    h_f, w_f, _, _ = self._w.shape
    pad_h = ((h_in - 1) * self._stride - h_f + 2) // 2
    pad_w = ((w_in - 1) * self._stride - w_f + 2) // 2
    return pad_h, pad_w

  def pad(self, array, pad):
    _, h, w, c = array.shape
    pad_h_before, pad_h_after = pad
    pad_w_before, pad_w_after = pad
    return np.pad(array, ((0, 0), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (0, 0)), mode='constant')

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
    n, h_in, w_in, _ = self._a_prev.shape
    h_f, w_f, _, _ = self._w.shape
    pad = self.calculate_pad_dims()
    a_prev_pad = self.pad(array=self._a_prev, pad=pad)
    output = np.zeros_like(a_prev_pad)

    self._db = da_curr.sum(axis=(0, 1, 2)) / n
    self._dw = np.zeros_like(self._w)

    for i in range(h_out):
      for j in range(w_out):
        h_start = i * self._stride
        h_end = h_start + h_f
        w_start = j * self._stride
        w_end = w_start + w_f
        # This part of the code performs element-wise multiplication between the flipped filters and the da_curr slice, 
        # then sums the result along the channel dimension.
        output[:, h_start:h_end, w_start:w_end, :] += np.sum(
            self._w[np.newaxis, :, :, :, :] * da_curr[:, i:i+1, j:j+1, np.newaxis, :],
            axis=4
        )
        # This part of the code updates the filter gradients by summing the element-wise product between the padded input and da_curr slice.
        self._dw += np.sum(
            a_prev_pad[:, h_start:h_end, w_start:w_end, :, np.newaxis] * da_curr[:, i:i+1, j:j+1, np.newaxis, :],
            axis=0
        )

    self._dw /= n
    return output[:, pad[0]:pad[0]+h_in, pad[1]:pad[1]+w_in, :]

# Example usage
a_prev = np.random.rand(2, 4, 4, 3)
w = np.random.rand(3, 3, 3, 2)
stride = 1
conv2d_backward = Conv2DBackward(a_prev, w, stride)
da_curr = np.random.rand(2, 2, 2, 2)
output = conv2d_backward.backward_pass(da_curr)
print(a_prev.shape)
print(output.shape)