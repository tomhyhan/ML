import os
import argparse
import yaml
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from linear_noise_scheduler import LinearNoiseScheduler
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from models.unet_cond import Unet
from models.vqvae import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config_value(config, key, default_value):
    return config[key] if key in config else default_value

def drop_class_condition(class_condition, class_drop_prob, im):
    if class_drop_prob > 0:
        class_drop_mask = torch.zeros((im.shape[0], 1), device=im.device).float().uniform_(0,
                                                                                           1) > class_drop_prob
        return class_condition * class_drop_mask
    else:
        return class_condition
    
def train(args):
    # SETTINGS
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Noise scheduler
    noise_scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    text_tokenizer = None
    text_model = None
    empty_text_embed = None
    condition_types = []
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)

    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']
        if "text" in condition_types:
            # later
            pass
    
    im_dataset_cls = {
        "mnist": MnistDataset,
        "celehq": CelebDataset
    }.get(dataset_config["name"])
    
    # path
    latent_path=os.path.join(train_config["task_name"], train_config["vqvae_latent_dir_name"])
    
    im_dataset = im_dataset_cls(split="train",
                              im_path=dataset_config["im_path"],
                              im_size=dataset_config["im_size"],
                              im_channels=dataset_config["im_channels"],
                              use_latents=True,
                              latent_path=latent_path,
                              condition_config=condition_config)

    data_loader = DataLoader(im_dataset, batch_size=train_config["ldm_batch_size"], shuffle=True)
    
    # init Unet model
    model = Unet(im_channels=autoencoder_model_config["z_channels"], model_config=diffusion_model_config).to(device)
    
    model.train()
    
    vae = None
    
    if not im_dataset.use_latents:
        vae = VQVAE(im_channels=dataset_config["im_channels"],
                    model_config=autoencoder_model_config).to(device)
        
        vae.eval()
        raise Exception("should not reach here")
    
    num_epochs = train_config["ldm_epochs"]
    optimizer = Adam(model.parameters(), lr=train_config["ldm_lr"])
    criterion = torch.nn.MSELoss()
    
    for epoch_idx in range(num_epochs):
        losses = []
        for data in tqdm(data_loader):
            cond_input = None
            if condition_config is not None:
                im, cond_input = data
            else:
                im = data
            optimizer.zero_grad()
            im = im.float().to(device)
            
            if "class" in condition_types:
                class_condition = torch.nn.functional.one_hot(
                    cond_input["class"],
                    condition_config["class_condition_config"]["num_classes"]
                ).to(device)
                class_drop_prob = get_config_value(condition_config["class_condition_config"], "cond_drop_prob", 0.)
                
                cond_input["class"] = drop_class_condition(class_condition, class_drop_prob, im)
            
            noise = torch.randn_like(im).to(device)
            
            t = torch.randint(0, diffusion_config["num_timesteps"], (im.shape[0],)).to(device)
            
            noisy_im = noise_scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t, cond_input=cond_input)
            loss =criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))
        
        check_pt_path = os.path.join(train_config['task_name'],train_config['ldm_ckpt_name'])
        
        torch.save(model.state_dict(), check_pt_path)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for training ldm")
    parser.add_argument("--config", dest="config_path", default="config/mnist.yaml", type=str)
    args= parser.parse_args()
    train(args)