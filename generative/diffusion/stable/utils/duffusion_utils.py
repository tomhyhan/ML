import pickle
import os
import torch
import glob

def load_latents(latent_path):
    latent_maps = {}
    for fname in glob.glob(os.path.join(latent_path, "*.pkl")):
        s = pickle.load(open(fname, 'rb'))
        for k, v in s.items():
            latent_maps[k] = v[0]
    return latent_maps
