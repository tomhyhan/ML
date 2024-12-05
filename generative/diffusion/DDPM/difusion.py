import torch
from torch import nn;
import torch.nn.functional as F

def extract(v,t,x_shape):
    # v[t]
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.size(0)] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(
        self,
        model,
        beta_l,
        beta_T,
        T
    ):
        super().__init__()
        self.model = model
        self.T = T
        
        self.register_buffer("betas", torch.linspace(beta_l, beta_T, T).double())
        
        alphas = 1 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0-alphas_bar))

    def forward(self, x0):
        t = torch.randint(self.T, size=(x0.size(0), )), device=x0.device
        noise = torch.randint_like(x0)
        xt = (extract(self.sqrt_alphas_bar, t, x0.shape) * x0 + extract(self.one_minus_sqrt_alphas_bar, t, x0.shape) * noise)
        loss = F.mse_loss(self.model(xt, t), noise, reduction='none')
        
        return loss
    
    
class GuassianDiffusionSampler(nn.Module):
    def __init__(
        self,
        model,
        beta_l,
        beta_T,
        T,
        img_size=32,
        mean_type="eps",
        var_type='fixedlarge'
    ):
        super().__init__()
        
        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        
        self.register("betas", torch.linspace(beta_l, beta_T, T))
        alphas = 1- self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, pad=[1,0], value=1)[:T] 