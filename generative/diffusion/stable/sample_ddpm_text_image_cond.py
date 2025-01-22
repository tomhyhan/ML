import os
import torch
import numpy as np
import argparse
import yaml
import torchvision
from transformers import DistilBertModel, DistilBertTokenizer, CLIPTokenizer, CLIPTextModel
from tqdm import tqdm

from models.vqvae import VQVAE
from linear_noise_scheduler import LinearNoiseScheduler
from models.unet_cond import Unet
from dataset.celeb_dataset import CelebDataset

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def get_tokenizer_and_model(model_type, device, eval_mode=True):
    if model_type == "bert":
        text_tokenizer = DistilBertTokenizer.from_pretrained("distilbert_base_uncased")
        text_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    else:
        text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16").to(device)

    if eval_mode:
        text_model.eval()

    return text_tokenizer, text_model 

def get_text_representation(text, text_tokenizer, text_model, device, truncation=True, padding="max_length", max_length=77):
    token_output = text_tokenizer(text, truncation=truncation, padding=padding, max_length=max_length)
    
    indexed_tokens = token_output["input_ids"]
    att_mask = token_output["attention_mask"]
    tokens_tensor = torch.tensor(indexed_tokens).to(device)
    mask_tensor = torch.tensor(att_mask).to(device)
    text_embed = text_model(tokens_tensor, attention_mask=mask_tensor).last_hidden_state
    return text_embed

def sample(
    model,
    scheduler,
    train_config,
    diffusion_model_config,
    autoencoder_model_config,
    diffusion_config,
    dataset_config,
    vae,
    text_tokenizer,
    text_model
):
    # define im_size
    im_size = dataset_config["im_size"] // 2 ** len(diffusion_model_config["down_sample"])
    
    # sample random noise latent
    xt = torch.randn(1, autoencoder_model_config["z_channel"], im_size, im_size)
    
    # create conditional input text_prompt and empty or neg text 
    text_prompt = ["she is a woman."]
    neg_prompt = ["he is a man."]
    empty_prompt = [""]
    
    text_prompt_embed = get_text_representation(text_prompt, text_tokenizer, text_model, device)
    empty_text_embed = get_text_representation(empty_prompt, text_tokenizer, text_model, device)
    
    # define condition config
    condition_config = diffusion_model_config["condition_config"]
    
    # define dataset
    dataset = CelebDataset(
        split="train",
        im_path=dataset_config["im_path"],
        im_size=dataset_config["im_size"],
        im_channels=dataset_config["im_channels"],
        use_latents=True,
        condition_config=condition_config
    ) 
    
    # get mask idx
    mask_idx = torch.randint(0, len(dataset.masks))
    # get mask
    mask = dataset.get_mask(mask_idx).unsqueeze(0).to(device)
    
    # define uncond_input and cond_input
    uncond_input = {
        "text": empty_text_embed,
        "mask": torch.zeros_like(mask)
    }
    
    cond_input = {
        "text": text_prompt_embed,
        "mask": mask
    }
    
    # define cf_guidance_scale
    cf_guidance_scale = train_config["cf_guidance_scale"]

    # loop through time steps
    for i in tqdm(reversed(range(diffusion_config["num_timesteps"]))):
        # define current time step
        T = torch.ones(xt.size(0)) * i        
        # get noise prediction from model 
        noise_pred_cond = model(xt, T, cond_input)
        
        # apply cf_guidance_scale if value is bigger than 1
        if cf_guidance_scale > 1:
            noise_pred_uncond = model(xt, T, uncond_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond 
        
        # get prev timestep and x0 prediction
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i))
        
        # save x0
        if i == 0:
            im = xt
        else:
            im = x0_pred
        
        
        
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
    
    # load text tokenizer and text model
    with torch.no_grad():
        text_tokenizer, text_model = get_tokenizer_and_model(condition_config["text_condition_config"]["text_embed_model"], device=device)
        
    # load unet model
    model = Unet(im_channels=autoencoder_model_config["z_channels"], model_config=diffusion_model_config).to(device)
    model.eval()
    
    ldm_ckpt_path = os.path.join(train_config["task_name"], train_config["ldm_ckpt_name"])
    if os.path.exists(ldm_ckpt_path):
        model.load_state_dict(torch.load(ldm_ckpt_path, map_location=device))
    
    # output dir
    outdir_path = train_config["task_name"]
    if not os.path.exists(outdir_path):
        os.mkdir(outdir_path)

    # load vqvae
    vae = VQVAE(im_channels=dataset_config["im_channels"], model_config=autoencoder_model_config).to(device)
    vae.eval()
    
    vae_ckpt_path = os.path.join(train_config["task_name"], train_config["vqvae_autoencoder_ckpt_name"])
    if os.path.exists(vae_ckpt_path):
        vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device))

    with torch.no_grad():
        sample(
            model,
            scheduler,
            train_config,
            diffusion_model_config,
            autoencoder_model_config,
            diffusion_config,
            dataset_config,
            vae,
            text_tokenizer,
            text_model
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample ddpm with text and image condition")
    parser.add_argument("--config", dest="config_path",default="data/config/celeb.yaml", type="str", )
    args = parser.parse_args()
    infer(args)
    
