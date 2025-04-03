import os
import yaml
import tqdm
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.voc import VOCDataset
from models.yolo import YOLOV1
from loss.yolov1_loss import YOLOV1Lloss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(data):
    return list(zip(*data))

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
    

    train_dataset = DataLoader(voc, 
                               shuffle=True, 
                               batch_size=train_config["batch_size"],
                               collate_fn=collate_fn
                               )
    
    yolo = YOLOV1(
        im_size=dataset_config["im_size"],
        num_classes=dataset_config["num_classes"],
        model_config=model_config
    )
    
    ckpt_path = os.path.join(train_config["task_name"]["ckpt_name"])
    if os.path.exists(ckpt_path):
        yolo.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    task_path = os.path.join(train_config["task_name"])
    if os.path.exists(task_path):
        os.mkdir(task_path)
    
    optimizer = torch.optim.SGD(
        params=filter(lambda p: p.requires_grad, yolo.parameters()),
        lr=train_config["lr"],
        weight_decay=5E-4,
        momentum=0.9
    )    
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, train_config["lr_steps"], gamma=0.5
    )
    
    criterion = YOLOV1Lloss()
    
    acc_steps = train_config["acc_steps"]
    num_epochs = train_config["num_epochs"]
    steps = 0
    
    for epoch_idx in range(num_epochs):
        losses = []
        optimizer.zero_grad()
        for idx, (ims, targets) in enumerate(tqdm(train_dataset)):
            yolo_targets = torch.cat([target["yolo_target"].unsqueeze(0).float().to(device) for target in targets], dim=0)
            im = torch.cat([im.unsqueeze(0).float().to(device) for im in ims], dim=0)
            pred = yolo(im)
            loss = criterion(pred, yolo_targets, use_sigmoid=model_config["use_sigmoid"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Arguments for training YOLOv1")
    parser.add_argument("--config", dest="config_path", default="config/voc.yaml", type=str)
    
    args, _ = parser.parse_known_args()
    
    train(args)
    pass