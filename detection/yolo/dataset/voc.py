import torch.nn as nn
from torch.utils.data import Dataset
import albumentations as A

class VOCDataset(Dataset):
    def __init__(self, split, im_sets, im_size=448, S=7, B=2, C=20):
        self.im_sets = im_sets
        self.fname = "trainval" if self.split == "train" else "test"
        self.im_size = im_size
        self.S = S
        self.B = B
        self.C = C
        
        self.transforms = {
            "train" : 
        }
        pass
