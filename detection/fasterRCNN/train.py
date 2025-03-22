import yaml
import torch
import random
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from dataset.voc import VOCDataset
from model.faster_rcnn import FasterRCNN

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
    np.random.seed(seed)
    random.seed(seed)

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

    fater_rcnn_model = FasterRCNN(
        model_config,
        num_classes=dataset_config["num_classes"]
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Arguments for training FasterRCNN")
    parser.add_argument("--config", dest="config_path",
                        default="config/voc.yaml", type=str)
    args = parser.parse_args()
    train(args)
