import os
import yaml
import torch
import numpy as np
import random
import torchvision
from torchvision.utils import make_grid
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.lpips import LPIPS
from ..models.discriminator import Discriminator
from vqvae import VQVAE
from ..dataset.celeb_dataset import CelebDataset
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    # read config file
    with open(args.config_path) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    
    # define configurable variables
    dataset_config = config["dataset_params"]
    autoencoder_config = config["autoencoder_params"]
    train_config = config["train_config"]
    
    # set random seed
    seed = train_config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        
    # create model 
    model = VQVAE(im_channels=dataset_config["im_channels"], config=autoencoder_config)
    
    # create dataset and data loader
    dataset = {
        "celebhq": CelebDataset
    }.get(dataset_config['name'])
    
    im_dataset = dataset(
        split="train",
        im_path=dataset_config["im_path"],
        im_size=dataset_config["im_size"],
        im_channels=dataset_config["im_channels"]
    )
    
    data_loader = DataLoader(im_dataset, batch_size=train_config['autoencoder_batch_size'], shuffle=True)
    
    # create output directory
    out_dir_path = dataset_config['name']
    if os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)
    
    # define num epochs
    num_epochs = train_config["autoencoder_epochs"]
    
    # define loss variables for reconstruction and Disc
    recon_loss_fn = nn.MSELoss()
    disc_loss_fn = nn.MSELoss()
    
    # define lpip model
    lpip_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels=dataset_config["im_channels"])
    
    # define opimizer
    g = torch.optim.Adam(model.parameters(), lr=train_config["autoencoder_lr"], betas=(0.5, 0.999))
    d = torch.optim.Adam(discriminator.parameters(), lr=train_config["autoencoder_lr"], betas=(0.5, 0.999))
    
    # define steps and disc start step
    disc_step_start = train_config["disc_start"]
    step_count = 0
        
    acc_steps = train_config["autoencoder_acc_steps"]
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0
    
    for epoch_idx in range(num_epochs):
        recon_losses = []
        codebook_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []

        for im in tqdm(data_loader):
            step_count += 1
            im = im.to(device)
            model_output = model(im)
            output, z, quant_losses = model_output

            if step_count == 1 or image_save_steps % step_count == 0:
                sample_size = min(8, im.size(0))
                sample_output = torch.clamp(output[:sample_size]).detach().cpu()
                sample_output = (sample_output + 1) / 2
                
                sample_input = im[:sample_size].detach().cpu()
                sample_input = (sample_input + 1) / 2
                
                samples = torch.concat([sample_input, sample_output], dim=0)
                grid = make_grid(samples, nrow=sample_size)
                sample_ims = torchvision.transforms.ToPILImage()(grid)
                image_path = os.path.join(train_config["task_name"], "vqvae_autoencoder_samples")
                sample_ims.save(image_path)
                sample_ims.close()
            
            ######### Optimize Generator ##########
            recon_loss = recon_loss_fn(output, im)
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            g_loss = recon_loss + (train_config["codebook_weight"] * quant_losses["codebook_loss"] / acc_steps) + (train_config["commitment_beta"] * quant_losses["commitment_loss"] / acc_steps)

            if step_count > disc_step_start:
                disc_fake_pred = discriminator(output)
                disc_loss = disc_loss_fn(disc_fake_pred, torch.ones_like(output))
                g_loss += train_config["disc_weight"] * disc_loss / acc_steps
            lpip_loss = torch.mean(lpip_model(output, im))
            g_loss += train_config["perceptual_weight"]*lpip_loss / acc_steps
            g_loss.backward()
            
            ######### Optimize Discriminator #######
            if step_count > disc_step_start:
                fake_out = discriminator(output.detach())
                fake_loss = discriminator(fake_out, torch.zeros_like(fake_out))
                real_loss = discriminator(im, torch.ones_like(im))
                d_loss = train_config["disc_weight"] * (fake_loss + real_loss) / 2
                d_loss = d_loss/ acc_steps
                d_loss.backword()
                if step_count % acc_steps == 0:
                    d.step()
                    d.zero_grad()
            if step_count % acc_steps == 0:
                g.step()
                g.zero_grad()
            d.step()
            d.zero_grad()
            g.step()
            g.zero_grad()
    
if __name__ == "__main__":
    # build argument parser
    parser = argparse.ArgumentParser(description="training vqvae")
    parser.add_argument("--config", dest="config_path",default="/data/celebhq.yaml", type="str")
    args = parser.parse_args()
    train(args)
