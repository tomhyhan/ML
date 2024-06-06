import torch

class Convolution:
    @staticmethod
    def forward(x, filter, b, padding, stride):
        """
            computes forward pass for the convolutional layer (faster version using pytorch Conv2d layer)
            
            Inputs:
                x:      (N, C_in, H, W) image input 
                filter: (C_out, C_in, H, W) filter size
                b:      (C_out,)
                pad:    extra padding applied to the image
                stride: number of stride 
        """
        N, C_in, H, W = x.shape
        C_out, C_in, HH, WW = filter.shape
        
        conv_2d = torch.nn.Conv2d(C_in,C_out,(HH,WW), stride=stride, padding=padding, device=x.device, dtype=x.dtype)
        
        conv_2d.weight = torch.nn.Parameter(filter)
        conv_2d.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        conv_x = conv_2d(tx)
        
        cache = (x, filter, b, padding, stride, tx, conv_x, conv_2d)
        return conv_x, cache
    
    @staticmethod
    def backward(dout, cache):
        """
            Computes backpropagation for convolutional layer using pytorch dynamic computatinal graph
            
            Inputs:
                dout: upstream gradients
                cache: input x, kernel, bias, padding, stride, input x with grad, result of conv2d on x, conv2d layer
            Outputs:
                dx: downstream gradients w.r.t. to x
                dw: gradients w.r.t. to w
                db: gradients w.r.t. to b
        """
        _, _, _, _, _, tx, conv_x, conv_2d = cache
        
        conv_x.backward(dout)
        dx = tx.grad.detach()
        dw = conv_2d.weight.grad.detach()
        db = conv_2d.bias.grad.detach()
        # reset gradients
        conv_2d.weight.grad = None
        conv_2d.bias.grad = None
        
        return dx, dw, db