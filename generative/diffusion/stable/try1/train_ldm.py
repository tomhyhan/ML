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

def get_text_representation(text, text_tokenizer, text_model, device, truncation=True, padding="max_length", max_length=77):
    token_output = text_tokenizer(
        text,
        truncation=truncation,
        padding=padding,
        return_attention_mask=True,
        max_length=max_length
    )
    
    indexed_tokens = token_output["input_ids"]
    attn_masks = token_output["attention_mask"]
    tokens_tensor = torch.tensor(indexed_tokens)
    mask_tensor = torch.tensor(attn_masks)
    text_embed = text_model(tokens_tensor, attn_masks=mask_tensor).last_hidden_state
    return text_embed
    
def get_tokenizer_and_model(model_type, device, eval_mode=True):
    if model_type == 'bert':
        text_tokenizer = DistilBertTokenizer.from_pretrained()
        text_model = DistilBertModel.from_pretrained()
    else:
        text_tokenizer = CLIPTextModel.from_pretrained()
        text_model = CLIPTextModel.from_pretrained()
        
    if eval_mode:
        text_model.eval()
    return text_tokenizer, text_model

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
    
    # noise scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config["num_timesteps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"]
    )
    
    text_tokenizer = None
    text_model = None
    empty_text_embed = None
    condition_config = diffusion_config["condition_config"]
    
    if condition_config is not None:
        condition_types = condition_config["condition_types"]
        if "text" in condition_types:
            with torch.no_grad():
                text_tokenizer, text_model = get_tokenizer_and_model(condition_config["text_condition_config"]["text_embed_model"], device=device)
                empty_text_embed = get_text_representation([""], text_tokenizer, text_model, device)
    
    im_dataset_cls = {
        "mnist" : MnistDataset,
        "celebhq" : CelebDataset
    }.get(dataset_config["name"])
    
    im_dataset = im_dataset_cls(
        split="train",
        im_path=dataset_config["im_path"],
        im_size=dataset_config["im_size"],
        im_channels=dataset_config["im_channels"],
        use_latents=True,
        latent_path=os.path.join(train_config["task_name"], train_config["vqvae_latent_dir_name"], condition_config=condition_config)
    )
    
    data_loader = DataLoader(im_dataset, batch_size=train_config["ldm_batch_size"], shuffle=True)
    
    model = Unet(im_channels=autoencoder_model_config['z_channels'], model_config=diffusion_config)
    model.train()
    
    
    
    
    
    
    
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/celebhq_text_cond_clip.yaml', type=str)
    args = parser.parse_args()
    train(args)