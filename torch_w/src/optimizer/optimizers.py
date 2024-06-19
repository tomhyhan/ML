import torch

def adam(w, dw, config: dict):
    """
        Adam Optimizer
    """
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
    m = config["m"]
    s = config["s"]
    eps = config["eps"]
    t = config["t"]    

    # print(w)
    # print(dw)
    t += 1
    m = (beta1 * m) + (1 - beta1) * dw
    s = (beta2 * s) + (1 - beta2) * torch.square(dw)
    
    mc = m / (1 - (beta1 ** t)) 
    sc = s / (1 - (beta2 ** t)) 
    
    w -= lr * mc / (torch.sqrt(sc)+ eps)
    
    config['m'] = m
    config['s'] = s
    config['t'] = t
    return config