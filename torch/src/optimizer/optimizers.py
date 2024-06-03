import torch
def adam(w, dw, config):
    """
        weight update with Adam update rule. 
        
        Inputs:
            w: weights
            dw: gradient w.r.t. w
            config:
                - learning rate: Scalar learning rate
                - beta1: Decay learning rate for momentum 
                - beta2: Decay learning rate for RMS prop
                - m: Moving average of momemtum 
                - s: Moving average of RMS
                - t: Iteration number
                - eps: small Contant for numeric stability  
    """
    if config is None:
        config = {}
    config.setdefault("learning rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("m", torch.zeros_like(w))
    config.setdefault("s", torch.zeros_like(w))
    config.setdefault("t", 0)
    config.setdefault("eps", 1e-8)
    pass