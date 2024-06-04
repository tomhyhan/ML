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
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("m", torch.zeros_like(w))
    config.setdefault("s", torch.zeros_like(w))
    config.setdefault("t", 0)
    config.setdefault("eps", 1e-8)
    
    lr = config["learning_rate"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    m = config['m']
    s = config['s']
    eps = config["eps"]
    
    config['t'] += 1
    t = config['t']
    
    
    m = (beta1 * m) + (1 - beta1) * dw 
    s = (beta2 * s) + (1 - beta2) * torch.square(dw)
    
    mc = m / (1 - (beta1 ** t))  
    sc = s / (1 - (beta2 ** t))  

    next_w = w - lr * mc / (torch.sqrt(sc) + eps)
    
    config['m'] = m
    config['s'] = s

    return next_w, config