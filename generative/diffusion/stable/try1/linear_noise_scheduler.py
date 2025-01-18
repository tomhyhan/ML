import torch
import torch.nn as nn

class LinearNoiseScheduler(nn.Module):
    def __init__(
        self,
        time_steps,
        beta_start,
        beta_end
    ):
        super().__init__()
        
        self.beta = torch.linspace(beta_end**0.5, beta_start**0.5, time_steps)**2
        
        self.alphas = 1 - self.beta
        
        self.alphas_cum_prod = torch.cumprod(self.alphas)
        self.sqrt_alphas_cum_prod = torch.sqrt(self.alphas_cum_prod)
        self.one_minus_cum_prod = 1 - self.alphas_cum_prod
        
    def add_noise(self, x0, noise, T):
        # = N (xt;√α¯tx0,(1 − α¯t)I) 
        sqrt_alphas_cum_prod_time = self.sqrt_alphas_cum_prod[T] 
        
        one_minus_cum_prod_time = self.one_minus_cum_prod[T]
        
        for _ in range(len(x0.shape)-1):
            sqrt_alphas_cum_prod_time.unsqueeze(-1)
        for _ in range(len(noise.shape)-1):
            one_minus_cum_prod_time.unsqueeze(-1)
        
        return sqrt_alphas_cum_prod_time * x0 + one_minus_cum_prod_time * noise

    def sample_prev_time_step(self, xt, noise_pred, T):
        # compute x0, mean, variance
        # µ˜t(xt, x0) := (√α¯t−1βt / 1 − α¯t)x0 + (√αt(1 − α¯t−1) / 1 − α¯txt) 
        # β˜t:= (1 − α¯t−1 / 1 − α¯t) βt
        pass