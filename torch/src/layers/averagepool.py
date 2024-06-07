from torch.nn import AdaptiveAvgPool2d
 
class AveragePool:
    @staticmethod
    def forward(X):
        """
            Compute forward pass for average pool layer. This is a Global Pool Layer that transforms the output of Conv layer into 1x1 grid.
            
            Inputs:
                X: (N, C_in, H, W) input data
            Outputs:
                out: (N, C_in, 1, 1)
        """
        
        avgpool = AdaptiveAvgPool2d((1,1))
        tx = X.detach()
        tx.requires_grad = True
        out_x = avgpool(tx)
        
        cache = (tx, out_x)
        return out_x, cache
    
    @staticmethod
    def backward(dout, cache):
        tx, out_x = cache 
        out_x.backward(dout)
        dx = tx.grad.detach()
        return dx
        
