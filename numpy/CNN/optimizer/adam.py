import numpy as np

class Adam:
    def __init__(self, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.dv = {}
        self.ds = {}        
        
    def update(self, layers, t=0):
        if len(self.dv) == 0 or len(self.ds) == 0:
            self.init_cache(layers)
        
        for idx, layer in enumerate(layers):
            gradients, weights = layer.gradients, layer.weights
            if gradients is None or weights is None:
                continue
            w, b = weights
            dw, db = gradients
            dw_key, db_key = self.get_cache_keys(idx)
            
            self.dv[dw_key] = self.beta1 * self.dv[dw_key] + (1 - self.beta1) * dw
            self.dv[db_key] = self.beta1 * self.dv[db_key] + (1 - self.beta1) * db
            
            self.ds[dw_key] = self.beta2 * self.ds[dw_key] + (1 - self.beta2) * np.square(dw)
            self.ds[db_key] = self.beta2 * self.ds[db_key] + (1 - self.beta2) * np.square(db)

            vdw_correct = self.dv[dw_key] / (1 - self.beta1**(t+1))
            vdb_correct = self.dv[db_key] / (1 - self.beta1**(t+1))
            sdw_correct = self.ds[dw_key] / (1 - self.beta2**(t+1))
            sdb_correct = self.ds[db_key] / (1 - self.beta2**(t+1))

            w = w - self.lr * vdw_correct / (np.sqrt(sdw_correct) + self.eps)
            b = b - self.lr * vdb_correct / (np.sqrt(sdb_correct) + self.eps)
            
            layer.set_weights(w, b)
            # later test with corrections



    def init_cache(self, layers):

        for idx, layer in enumerate(layers):
            gradients = layer.gradients 
            if gradients is None:
                continue
            
            dw, db = gradients
            dw_key, db_key = self.get_cache_keys(idx)
            # print("asdf")
            self.dv[dw_key] = np.zeros_like(dw)
            self.dv[db_key] = np.zeros_like(db)
            self.ds[dw_key] = np.zeros_like(dw)
            self.ds[db_key] = np.zeros_like(db)

    @staticmethod
    def get_cache_keys(idx):
        return f"dw{idx}", f"db{idx}"
    