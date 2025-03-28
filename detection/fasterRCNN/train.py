import os
import torch.utils
import yaml
import torch
import random
import numpy as np
import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from dataset.voc import VOCDataset
from model.faster_rcnn import FasterRCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    dataset_config = config["dataset_params"]
    model_config = config["model_params"]
    train_config = config["train_params"]

    seed = train_config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    
    voc = VOCDataset(
        "train",
        img_dir=dataset_config["im_train_path"],
        ann_dir=dataset_config["ann_train_path"]
    )

    train_dataset = DataLoader(
        voc,
        batch_size=1,
        shuffle=True,
        num_workers=4
    )

    faster_rcnn_model = FasterRCNN(
        model_config,
        num_classes=dataset_config["num_classes"]
    )
    faster_rcnn_model.train()
    faster_rcnn_model.to(device)
    
    if not os.path.exists(train_config["task_name"]):
        os.mkdir(train_config["task_name"])
        
    optimizer = torch.optim.SGD(
        lr=train_config["lr"],
        params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
        weight_decay=5e-4,
        momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=train_config["lr_steps"], gamma=0
    )
    
    acc_steps = train_config["acc_steps"]
    num_epochs = train_config["num_epochs"]
    step_count = 1
    
    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        optimizer.zero_grad()
        
        for im, target, fname in tqdm(train_dataset):
            target["labels"] =  target["labels"].long().to(device)
            target["bboxes"] =  target["bboxes"].float().to(device)
            rpn_output, frcnn_output = faster_rcnn_model(im, target)
            
            rpn_loss = rpn_output["rpn_classification_loss"] + rpn_output["rpn_localization_loss"]
            frcnn_loss = frcnn_output["frcnn_classification_loss"] + frcnn_output["frcnn_localization_loss"]
            loss = rpn_loss + frcnn_loss
            
            rpn_classification_losses.append(rpn_output["rpn_classification_loss"].item())
            rpn_localization_losses.append(rpn_output["rpn_localization_loss"].item())
            frcnn_classification_losses.append(frcnn_output["frcnn_classification_loss"].item())
            frcnn_localization_losses.append(frcnn_output["frcnn_localization_loss"].item())

            loss = loss / acc_steps
            loss.backward()
            
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_count += 1
        print(f"Finished epoch {i+1}")
        optimizer.step()
        optimizer.zero_grad()
        torch.save(faster_rcnn_model.parameters(), os.path.join(train_config["task_name"], train_config["ckpt_name"]))
        
        scheduler.step()
    print("Done training")

if __name__ == "__main__":
    parser = ArgumentParser(description="Arguments for training FasterRCNN")
    parser.add_argument("--config", dest="config_path",
                        default="config/voc.yaml", type=str)
    args = parser.parse_args()
    train(args)
