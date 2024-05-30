import torch


class Maxpool:

    @staticmethod
    def forward(x):
        """
            Computes forward pass for maxpool layer. Maxpool always always shrinks the image size by halves 

            Filter size: (2,2)
            stride: 2

        Inputs:
            x: (N, C_in, H, W) image input
        Outputs:
            max_x: max_pool of x
            cache: values use for backward
        """
        maxpool = torch.nn.MaxPool2d((2, 2), 2)
        tx = x.detach()
        tx.requires_grad = True
        max_x = maxpool(tx)
        
        cache = (tx, max_x)
        return max_x, cache

    def backward(dout, cache):
        """
            Computes backward pass for maxpoll layer.

            Inputs:
                dout: upstream gradients 
                cahce: input x, result of maxpooled x
            Outputs:
                dx: downsteam gradients w.r.t. x
        """
        tx, max_x = cache
        max_x.backward(dout)
        dx = tx.grad.detach()
        return dx
