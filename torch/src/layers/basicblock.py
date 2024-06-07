from .conv import Convolution
from .relu import ReLU
from .batch_norm import BatchNorm
from .identity import Identity

class BasicBlock:
    @staticmethod
    def forward(X, weights, biases, gammas, betas, bn_params, stride, down_sample=False):
        """
            Computes forward pass for resnet basic block 
            
            Inputs:
                X: (N, C_in, H, W) data input
                weight: (C_out, C_in, HH, WW) filters for two conv layers
                biases: biases for two conv layers
                stride: stride to be applied to the first conv layer. When stride is > 1, it down samples the input data
                gammas: gammas for two bn layers 
                betas: betas for two bn layers 
                bn_params: bn_params for two bn layers 
            Outputs:
                out: output from basicblock
        """
        padding = 1
        if down_sample:
            w1, w2, w3 = weights
            b1, b2, b3 = biases
            gamma1, gamma2, gamma3 = gammas
            beta1, beta2, beta3= betas
            bn_param1, bn_param2, bn_param3 = bn_params
        else:
            w1, w2 = weights
            b1, b2 = biases
            gamma1, gamma2 = gammas
            beta1, beta2= betas
            bn_param1, bn_param2 = bn_params
        
        out_identity = Identity.forward(X)
        
        out, conv_cache1 = Convolution.forward(X, w1, b1, padding=padding,stride=stride)
        out, bn_cache1 = BatchNorm.forward(out, gamma1, beta1, bn_param1)
        out, relu_cache1 = ReLU.forward(out)
        out, conv_cache2 = Convolution.forward(out, w2, b2, padding=padding,stride=1)
        out, bn_cache2 = BatchNorm.forward(out, gamma2, beta2, bn_param2)
        
        conv_cache3 = bn_cache3 = None
        if down_sample:
            out_identity, conv_cache3 = Convolution.forward(out_identity, w3, b3, stride=2, padding=0)
            out_identity, bn_cache3 = BatchNorm.forward(out_identity, gamma3, beta3, bn_param3)   

        out += out_identity
        out, relu_cache2 = ReLU.forward(out)

        cache = (conv_cache1, bn_cache1, relu_cache1, conv_cache2, bn_cache2, relu_cache2, conv_cache3, bn_cache3)

        return out, cache
    
    @staticmethod
    def backward(dout, cache):
        conv_cache1, bn_cache1, relu_cache1, conv_cache2, bn_cache2, relu_cache2, conv_cache3, bn_cache3 = cache
        
        dout = ReLU.backward(dout, relu_cache2)
        didentity = dout.clone()

        dw3 = db3 = dgamma3 = dbeta3 = None
        if conv_cache3 is not None:
            didentity, dw3, db3 = BatchNorm.backward(didentity, bn_cache3)
            didentity, dgamma3, dbeta3= Convolution.backward(didentity, conv_cache3)
        dout, dgamma2, dbeta2 = BatchNorm.backward(dout, bn_cache2)        
        dout, dw2, db2 = Convolution.backward(dout, conv_cache2)        
        dout = ReLU.backward(dout, relu_cache1)
        dout, dgamma1, dbeta1 = BatchNorm.backward(dout, bn_cache1)        
        dout, dw1, db1 = Convolution.backward(dout, conv_cache1)        
        
        dout += didentity
        
        dweights = [dw1, dw2]
        dbiases = [db1, db2]
        dgamma = [dgamma1, dgamma2]
        dbeta = [dbeta1, dbeta2]
        
        if conv_cache3 is not None:
            dweights.append(dw3)
            dbiases.append(db3)
            dgamma.append(dgamma3)
            dbeta.append(dbeta3)
        
        # grads = {
        #     "dw3": dw3, 
        #     "db3": db3, 
        #     "dgamma3": dgamma3, 
        #     "dbeta3": dbeta3, 
        #     "dgamma2": dgamma2, 
        #     "dbeta2": dbeta2, 
        #     "dw2": dw2, 
        #     "db2": db2, 
        #     "dgamma1": dgamma1, 
        #     "dbeta1": dbeta1, 
        #     "dw1": dw1, 
        #     "db1": db1, 
        # }
        return dout, dweights, dbiases, dgamma, dbeta