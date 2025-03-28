import os
import yaml
import random
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset.voc import VOCDataset
from model.faster_rcnn import FasterRCNN

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def load_model_and_dataset(args):
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    
    # define configs
    dataset_config = config["dataset_params"]
    model_config = config["model_params"]
    train_config = config["train_params"]
    
    # seed everything
    seed = train_config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    # load test dataset
    voc = VOCDataset(
        "test", 
        dataset_config["im_test_path"],
        dataset_config["ann_testPath"]
    )
    
    # load data loader
    test_dataset = DataLoader(voc, batch_size=1, shuffle=True)
    
    # init FasterRCNN model
    faster_rcnn_model = FasterRCNN(
        model_config,
        dataset_config["num_classes"]
    )
    
    # load the params
    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    faster_rcnn_model.load_state_dict(
        torch.load(
            os.path.join(train_config["voc"], train_config["ckpt_name"]),
            map_location=device
        )
    )
    
    return faster_rcnn_model, voc, test_dataset
        

def infer(args):
    sample_path = "samples"
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)
        
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset(args)
    
    faster_rcnn_model.roi_head.low_score_threshold = 0.7


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for faster cnn inference")
    parser.add_argument("--config",default="config/yaml", dest="config_path", type=str)
    parser.add_argument("--infer_samples", dest="infer_samples", default=True, type=bool)
    
    args, _ = parser.parse_known_args()
    infer(args)