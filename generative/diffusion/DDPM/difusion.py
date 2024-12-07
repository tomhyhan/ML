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
        t = torch.randint(self.T, size=(x0.size(0), ), device=x0.device) 
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
        
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1.0/alphas_bar)
        )
        self.register_buffer(
            'sqrt_recip_m1_alphas_bar', torch.sqrt(1.0/alphas_bar - 1)
        )
        
        self.register_buffer(
            'posterior_var', self.beta * (1.0-alphas_bar_prev) / (1.0-alphas_bar)
        )
        
        self.register_buffer(
            'posterior_log_var_clipped', torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]]))
        )
        self.register_buffer(
            'posterior_mean_coef1', torch.sqrt(alphas_bar_prev) * self.betas / (1.0 - alphas_bar)
        )
        self.register_buffer(
            'posterior_mean_coef2', torch.sqrt(alphas) * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        )
        
        
    def predict_xstart_from_eps(self, xt, t, eps):
        return extract(self.sqrt_recip_alphas_bar, t, xt.shape) * xt - extract(self.sqrt_recipmi_alphas_bar, t, xt.shape) * eps
    
    def q_mean_variance(self, x0, xt, t):
        posterior_mean = extract(self.posterior_mean_ceof1, t, xt.shape) * x0 + extract(self.posterior_mean_ceof2, t, xt.shape) * xt 

        posterior_log_var_clipped = extract(self.posterior_log_var_clipped, t, xt.shape) 
        
        return posterior_mean ,posterior_log_var_clipped 
    
    def p_mean_variance(self, xt, t):
        model_log_var = {
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2], self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped
        }[self.var_type]
        model_log_var = extract(model_log_var, t, xt.shape)
        if self.mean_type == "epsilon":
            eps = self.model(xt, t)
            x0 = self.predict_xstart_from_eps(xt, t, eps)
            model_mean, _ = self.q_mean_variance(x0, xt, t)
        x0 = torch.clip(x0, -1, 1)

        return model_mean, model_log_var
    
    def forward(self, xT):
        xt = xT
        for time_step in reversed(range(self.T)):
            t = xt.new_ones([xT.size(0),], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(xt=xt, t=t)
            if time_step > 0:
                noise = torch.randn_like(xt)
            else:
                noise = 0
                
            xt = mean + torch.exp(0.5 * log_var) * noise
        x0 = xt
        return torch.clip(x0, -1, 1)
        
        