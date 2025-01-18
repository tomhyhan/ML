import os
import argparse
import yaml
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer, CLIPTokenizer, CLIPTextModel

from linear_noise_scheduler import LinearNoiseScheduler
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from models.unet_cond import Unet
from models.vqvae import VQVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/celebhq_text_cond_clip.yaml', type=str)
    args = parser.parse_args()
    train(args)