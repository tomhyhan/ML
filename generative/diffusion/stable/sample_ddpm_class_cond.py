import os
import yaml
import torch
import argparse

from linear_noise_scheduler import LinearNoiseScheduler
from models.unet_cond import Unet
from models.vqvae import VQVAE

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def get_config_value(config, key, default_value):
    return config[key] if key in config else default_value

def sample(
    model,
    scheduler,
    train_config,
    diffusion_model_config,
    autoencoder_model_config,
    diffusion_config,
    dataset_config,
    vae
):
    # z size
    im_size = dataset_config["im_size"] // (2**sum(autoencoder_model_config["down_sample"]))
    
    # sample random noise 
    xt = torch.randn((train_config["num_samples"], autoencoder_model_config["z_channels"], im_size, im_size))
    
    condition_config = get_config_value(diffusion_model_config, key="condition_config", default_value=None)
    condition_types = get_config_value(condition_config, condition_types, []) 
    
    # create conditional input
    num_classes = condition_config["class_condition_config"]["num_classes"]
    sample_classes = torch.randint(0,)

def infer(args):
    with open(args.config_path) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
            
    diffusion_config = config["diffusion_params"]
    dataset_config = config["dataset_params"]
    diffusion_model_config = config["ldm_params"]
    autoencoder_model_config = config["autoencoder_params"]
    train_config = config["train_params"]

    # load noise cheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config["num_timesteps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
    )
    
    # load unet model
    model = Unet(im_channels=autoencoder_model_config["z_channels"], model_config=diffusion_model_config).to(device)

    model.eval()
    unet_ckpt_path = os.path.join(train_config["task_name"], train_config["ldm_ckpt_name"])
    
    if os.path.exists(unet_ckpt_path):
        print("loaded unet checkpoint")
        model.load_state_dict(torch.load(unet_ckpt_path, map_location=device))
    else:
        raise Exception(f"model checkpoint {unet_ckpt_path} not found")

    task_path = os.path.join(train_config["task_name"])
    
    if not os.path.exists(task_path):
        os.mkdir(task_path)
        
    # load vae model
    vae = VQVAE(im_channels=dataset_config["im_channels"], model_config=autoencoder_model_config)
    
    vae.eval()
    
    vae_path = os.path.join(train_config["task_name"], train_config["vqvae_autoencoder_cpkt_name"])
    
    if os.path.exists(vae_path):
        print("loaded vae checkpoint")
        vae.load_state_dict(torch.load(vae_path, map_location=device), strict=True)
    else:
        raise Exception(f"VAE checkpoint {vae_path} not found")
    
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config, autoencoder_model_config, diffusion_config, dataset_config, vae)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Arguments for ddpm image generation for class conditional")
    parse.add_argument("--config", dest="config_path", default="config/mnist.yaml", type=str)
    args = parse.parse_args()
    infer(args)