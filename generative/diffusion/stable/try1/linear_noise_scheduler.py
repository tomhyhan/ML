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
        
        self.betas = torch.linspace(beta_end**0.5, beta_start**0.5, time_steps)**2
        
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
        
        # (xt - (1 − α¯t)I) / √α¯t = x0
        x0 = (xt - self.one_minus_cum_prod[T]*noise_pred) / torch.sqrt
        (self.sqrt_alphas_cum_prod[T])
        x0 = torch.clamp(x0, -1., 1.)
        
        mean = (xt - (self.betas[T] / torch.sqrt(self.one_minus_cum_prod[T]))) / torch.sqrt(self.alphas_cum_prod)

        if T == 0:
            return mean, x0
        else:
            # in paper
            # variance = self.betas[T]
            
            variance = (1 - self.alphas_cum_prod[T-1]) / (1 - self.alphas_cum_prod[T]) * self.betas[T]
            sigma = variance ** 0.5
            z = torch.randn_like(xt)
            
            return mean + z * sigma, x0