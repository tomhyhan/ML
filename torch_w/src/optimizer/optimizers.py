def adam(w, dw, config: dict):
    
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    pass