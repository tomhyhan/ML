import torch

beta_start = 0.2
beta_end = 0.99
num_timesteps = 10
x = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
print(x[1] - x[0], x[2] - x[1], x[3] - x[2] )
print(x)