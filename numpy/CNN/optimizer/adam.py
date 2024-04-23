import numpy as np

class Adam:
    def __init__(self, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.cache_v = {}
        self.cache_s = {}
        
    def update(self, layers, t):
        if len(self.cache_s) == 0 or len(self.cache_v) == 0:
            self.init_cache(layers)
        
        for idx, layer in enumerate(layers):
            weights, gradients = layer.weights, layer.gradients
            
            if weights is None or gradients is None:
                continue
            
            (w,b), (dw, db) = weights, gradients
            dw_key, db_key = Adam.get_cache_key(idx)
            
            self.cache_v[dw_key] = self.beta1 * self.cache_v[dw_key] + (1 - self.beta1) * dw 
            self.cache_s[db_key] = self.beta1 * self.cache_s[db_key] + (1 - self.beta1) * db 

            self.cache_s[dw_key] = self.beta2 * self.cache_s[dw_key] + (1 - self.beta2) * np.square(dw)
            self.cache_s[db_key] = self.beta2 * self.cache_s[db_key] + (1 - self.beta2) * np.square(db)
            
            w = w  - self.lr * self.cache_v[dw_key] / np.sqrt(self.cache_s[dw_key] + self.eps)
            b = b - self.lr * self.cache_s[db_key] / np.sqrt(self.cache_s[db_key] + self.eps)
            
            layer.set_weights(w, b)
            
            # later test with corrections
            # vdw_correct = self.cache_v[dw_key] / (1 - self.beta1**t)
            # vdb_correct = self.cache_v[db_key] / (1 - self.beta1**t)
            # sdw_correct = self.cache_s[dw_key] / (1 - self.beta2**t)
            # sdb_correct = self.cache_s[db_key] / (1 - self.beta2**t)



    def init_cache(self, layers):
        for idx, layer in enumerate(layers):
            gradients = layer.gradients
            if gradients is None:
                continue
            
            dw, db = gradients
            dw_key, db_key = Adam.get_cache_key(idx)
            
            self.cache_v[dw_key] = np.zeros_like(dw)
            self.cache_v[db_key] = np.zeros_like(db)
            self.cache_s[dw_key] = np.zeros_like(dw)
            self.cache_s[db_key] = np.zeros_like(db)

    @staticmethod
    def get_cache_key(idx):
        return f"dw{idx}", f"db{idx}"
    