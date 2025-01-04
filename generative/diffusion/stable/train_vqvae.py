import yaml
import argparse
from easydict import EasyDict
import numpy as np
import torch
import random
from models.vqvae import VQVAE
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
    
    im_dataset_cls = {
        "mnist" : MnistDataset,
        "celebhq" : CelebDataset
    }.get(dataset_config["name"])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for training VQVAE")
    parser.add_argument("--config", dest="config_path",  default="config/mnist.yaml", type=str)
    args = parser.parse_args()
    train(args)