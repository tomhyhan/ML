from conv import Convolution
from batch_norm import BatchNorm
from relu import ReLU
from maxpool import Maxpool

class SequentialConv:
    
    @staticmethod
    def forward(X,w,b,gamma,beta, bn_param):
        """
            Computes forward pass for 
            conv - bn - relu - maxpool
            
            Inputs:
                X: input data
            outputs:
                out: output of final layer
        """
        padding = 1
        stride = 1
        out, conv_cache = Convolution.forward(X,w,b,padding, stride)
        out, bn_cache = BatchNorm.forward(out, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(out)
        out, pool_cache = Maxpool.forward(out)
        
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache


    @staticmethod
    def backward(dout,cache):
        """
            computes backward pass for:
                pool - relu - bn - conv
                
            Inputs:
                dout: upstream gradients input
                cache: values for backward pass
            Outputs:
                dout: gradients w.r.t. x
                dw: gradients w.r.t. w
                db: gradients w.r.t. b
                dgamma: gradients w.r.t. gamma
                dbeta: gradients w.r.t. beta
        """
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        
        dout = Maxpool.backward(dout, pool_cache)
        dout = ReLU.backward(dout, relu_cache)
        dout, dgamma, dbeta = BatchNorm.backward(dout, bn_cache)
        dout, dw, db = Convolution.backward(dout, conv_cache)
        
        return dout, dw, db, dgamma, dbeta
        
        