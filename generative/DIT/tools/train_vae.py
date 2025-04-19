import yaml
import random
import argparse
import torch
import  numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    config_path = args.config_path
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    dataset_config = config["dataset_params"]
    autoencoder_config = config["autoencoder_params"]
    train_config = config["train_params"]

    seed = train_config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        
    # create the model and dataset
    model = VAE(
        im_channels=dataset_config["im_channels"],
        model_config=autoencoder_config
    ).to(device)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for training VAE")
    parser.add_argument("--config", dest="config_path", default="config/celebhq.yaml", type=str)
    args = parser.parse_args()
    train(args)