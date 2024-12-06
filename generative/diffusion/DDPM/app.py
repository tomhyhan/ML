# Sample params from paper
# T = 1000
# m β1 = 10−4 to βT = 0.02
import torch
from tqdm import trange
from load_data import load_cifar
from copy import deepcopy

from model import Unet
from difusion import GaussianDiffusionTrainer

def infinite_loop(loader):
    while True:
        for batch in loader:
            yield batch


# Params
steps = 10
batch_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-4
warmup=5000

train_loader = load_cifar(batch_size)
infinite_loader = infinite_loop(train_loader)
T=1000
ch=128
attn = [1]
beta_l = 1e-4
beta_T = 0.02
mean_type = "epsilon"
var_type = "fixedlarge"
# model setup
# [1, 2, 2, 2]
model = Unet(time_steps=T, channels=ch, channel_mult=[1, 2, 2, 2], attn=True, n_res_blocks=2, dropout=0.1)
ema_model = deepcopy(model)
trainer = GaussianDiffusionTrainer(model, beta_l, beta_T, T)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda step: min(step, warmup)/ warmup)

with trange(steps, dynamic_ncols=True) as pbar:
    for step in pbar:
        pass