import numpy as np

class Adam:
    def __init__(self, lr, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.dv = {}
        self.ds = {}        
        
    def update(self, layers, t=None):
        if len(self.dv) == 0 or len(self.ds) == 0:
            self.init_cache()
        
        
        for idx, layer in enumerate(layer):
            if layer.gradients is None:
                continue
            
            dw, db = layer.gradients
            dw_key, db_key = self.get_cache_keys(idx)
            
            vdw = self.beta1 * self.dv[dw_key] + (1 - self.beta1) * dw
            # later test with corrections
            # vdw_correct = self.cache_v[dw_key] / (1 - self.beta1**t)
            # vdb_correct = self.cache_v[db_key] / (1 - self.beta1**t)
            # sdw_correct = self.cache_s[dw_key] / (1 - self.beta2**t)
            # sdb_correct = self.cache_s[db_key] / (1 - self.beta2**t)



    def init_cache(self, layers):

        for idx, layer in enumerate(layers):
            if layer.gradients is None:
                return None
            
            dw, db = layer.gradients
            dw_key, db_key = self.get_cache_keys(idx)

            self.dv[dw_key] = np.zeros_like(dw)
            self.dv[db_key] = np.zeros_like(db)
            self.ds[dw_key] = np.zeros_like(dw)
            self.ds[db_key] = np.zeros_like(db)

    @staticmethod
    def get_cache_keys(idx):
        return f"dw{idx}", f"db{idx}"
    