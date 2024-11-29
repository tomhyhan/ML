# Sample params from paper
# T = 1000
# m β1 = 10−4 to βT = 0.02
from tqdm import trange
from time import sleep
from load_data import load_cifar

from model import Unet
def infinite_loop(loader):
    while True:
        for batch in loader:
            yield batch


# Params
steps = 10
batch_size = 128


train_loader = load_cifar(batch_size)
infinite_loader = infinite_loop(train_loader)
T=1000
ch=128
attn = [1]
# model setup
# [1, 2, 2, 2]
model = Unet(time_steps=T, channels=ch, channel_mult=[1, 2, 2, 2], attn=True, n_res_blocks=2, dropout=0.1)

with trange(steps, dynamic_ncols=True) as pbar:
    for step in pbar:
        pass