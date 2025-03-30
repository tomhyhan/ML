import os
import yaml
import random
import argparse
import numpy as np
import torch

from dataset.voc import VOCDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    config_path = args.config_path
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    dataset_config = config["dataset_params"]
    model_config = config["model_params"]
    train_config = config["train_params"]
    
    seed = train_config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    voc = VOCDataset(
        "train",
        im_sets=dataset_config["train_im_sets"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Arguments for training YOLOv1")
    parser.add_argument("--config", dest="config_file", default="config/voc.yaml", type=str)
    
    args, _ = parser.parse_known_args()
    
    train(args)
    pass