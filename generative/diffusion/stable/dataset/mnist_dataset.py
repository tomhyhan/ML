
import os
import glob
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from utils.diffusion_utils import load_latents

# import sys
# sys

class MnistDataset(Dataset):
    def __init__(self, 
        split,
        im_path,
        im_size,
        im_channels,
        use_latents=False,
        latent_path=None,
        condition_config=None
    ):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        
        self.latent_maps = None
        self.use_latents = False
        
        self.condition_types = [] if condition_config is None else condition_config["condition_types"]

        self.images, self.labels = self.load_images(im_path)
        
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print(f"Found latents {len(self.latent_maps)}")
            else:
                print("latents not found")
                
    def load_images(self, im_path):
        ims = []
        labels = []
        
        for d_name in tqdm(os.listdir(im_path)):
            fnames = glob.glob(os.path.join(im_path, d_name, '*.png'))
            fnames += glob.glob(os.path.join(im_path, d_name, '*.jpg'))
            fnames += glob.glob(os.path.join(im_path, d_name, '*.jpeg'))
            for fname in fnames:
                ims.append(fname)
                if "class" in self.condition_types:
                    labels.append(int(d_name))
        print(f"Found {len(ims)} for split {self.split}")
        return ims, labels

    def __len__(self):
        return len(self.images)

    