import torch 

class BatchNorm:

    @staticmethod
    def forward(x, gamma, beta, bn_param={}, mode="train", eps=1e-5, momentum=0.1):
        """
            Compute forward pass for Batch Norm
            
            Inputs:
                x: (N, C_in, H, W) input data
                gamma: (C_in,) scale parameter
                beta: (C_in,) shift parameter
                mode: "train" or "test"
                eps: constant for numeric stability
                momentum: constant for running mean and variance
            Outputs:
                batchnormx: (N, C_in, H, W) the result of batch norm on x.
                cache: values needed for backward pass
        """
        C = x.shape[1]
        if "running_mean" not in bn_param:
            bn_param["running_mean"] = torch.zeros(C, device=x.device, dtype=x.dtype) 
            bn_param["running_var"] = torch.ones(C, device=x.device, dtype=x.dtype)  

        batchnorm_layer = torch.nn.BatchNorm2d(C, eps, momentum, device=x.device, dtype=x.dtype)

        batchnorm_layer.weight = torch.nn.Parameter(gamma)
        batchnorm_layer.bias = torch.nn.Parameter(beta)
        batchnorm_layer.running_mean = torch.nn.Parameter(bn_param["running_mean"], requires_grad=False)
        batchnorm_layer.running_var = torch.nn.Parameter(bn_param["running_var"], requires_grad=False)
        
        if mode == "train":
            batchnorm_layer.train()
        elif mode == "test":
            batchnorm_layer.eval()
            
        tx = x.detach()
        tx.requires_grad = True
        batchnormx = batchnorm_layer(tx)
        
        cache = (tx, batchnormx, batchnorm_layer)
        return batchnormx, cache
    
    @staticmethod
    def backward(dout, cache):
        """
            Computes backward pass for Batch norm layer
            
            Inputs:
                dout: upstream gradient same size as input x
                cache: 
                    tx: input x
                    batchnormx: the result of computing B.N. on x
                    batchnorm_layer: torch implemetatino of B.N.
            Outpus:
                dx: gradients w.r.t. x
                dgamma: gradients w.r.t. gamma
                dbeta: gradients w.r.t. beta
        """
        (tx, batchnormx, batchnorm_layer) = cache
        batchnormx.backward(dout, retain_graph=True)
        dx = tx.grad.detach()
        dgamma = batchnorm_layer.weight.grad.detach()
        dbeta = batchnorm_layer.bias.grad.detach()
        batchnorm_layer.weight.grad = None
        batchnorm_layer.weight.bias = None
        
        return dx, dgamma, dbeta
        
