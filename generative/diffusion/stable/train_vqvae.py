import os
import yaml
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from easydict import EasyDict

from models.vqvae import VQVAE
from models.lpips import LPIPS
from models.discriminator import Discriminator
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def train(args):
    
    with open(args.config_path) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    config = EasyDict(config)

    dataset_config = config.dataset_params
    autoencoder_config = config.autoencoder_params
    train_config = config.train_params
    
    seed = train_config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        
    model = VQVAE(
        im_channels=dataset_config.im_channels,
        model_config=autoencoder_config
    ).to(device)
    
    im_dataset_cls: MnistDataset= {
        "mnist" : MnistDataset,
        "celebhq" : CelebDataset
    }.get(dataset_config["name"])

    im_dataset = im_dataset_cls(
        split="train",
        im_path=dataset_config["im_path"],
        im_size=dataset_config["im_size"],
        im_channels=dataset_config["im_channels"]
    )
    
    data_loader = DataLoader(
        im_dataset,
        batch_size=train_config["autoencoder_batch_size"],
        shuffle=True
    )
    
    if not os.path.exists(train_config["task_name"]):
        os.mkdir(train_config["task_name"])
        
    recon_criterion = torch.nn.MSELoss()
    disc_criterion = torch.nn.MSELoss()
    
    lpips_model = LPIPS().eval().to(device)
    print('dataset_config["im_channels"]', dataset_config["im_channels"])
    
    discriminator = Discriminator(im_channels=dataset_config["im_channels"]).to(device)
    
    optimizer_d = Adam(discriminator.parameters(), lr=train_config["autoencoder_lr"], beta=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config["autoencoder_lr"], beta=(0.5, 0.999))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for training VQVAE")
    parser.add_argument("--config", dest="config_path",  default="config/mnist.yaml", type=str)
    args = parser.parse_args()
    train(args)