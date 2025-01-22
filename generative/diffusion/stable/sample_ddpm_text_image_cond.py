import torch
import numpy as np
import argparse
import yaml

from linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def sample():
    pass

def infer(args):
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    diffusion_config = config["diffusion_params"]
    dataset_config = config["dataset_params"]
    diffusion_model_config = config["ldm_params"]
    autoencoder_model_config = config["autoencoder_params"]
    train_config = config["train_params"]

    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config["num_timesteps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"]
    )    
    
    condition_config = diffusion_model_config["condition_config"]
    condition_types = condition_config["condition_types"]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample ddpm with text and image condition")
    parser.add_argument("--config", dest="config_path",default="data/config/celeb.yaml", type="str", )
    args = parser.parse_args()
    infer(args)
    
