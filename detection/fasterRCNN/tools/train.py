import yaml
import torch
import random
import numpy as np
from argparse import ArgumentParser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    print(args)
    
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
                
    dataset_config = config["dataset_params"]
    model_config = config["model_params"]
    train_config = config["train_params"]
    
    seed = train_config["seed"]
    torch.manual_seed(seed)
    np.random(seed)
    random.seed(seed)
    
    voc = VOCDataset(
                "train",
                im_dir=dataset_config["im_train_path"],
                ann_dir=dataset_config["ann_train_path"]
            )
    
        
if __name__ == "__main__":
    parser = ArgumentParser(description="Arguments for training FasterRCNN")
    parser.add_argument("--config", dest="config_path", default="config/voc.yaml", type=str)
    args = parser.parse_args()
    train(args)
    
    