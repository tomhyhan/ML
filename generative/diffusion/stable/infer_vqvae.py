import os
import yaml
import glob
import pickle
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid
from argparse import ArgumentParser
from tqdm import tqdm

from models.vqvae import VQVAE
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer(args):

    with open(args.config_path) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

    dataset_config = config["dataset_params"]
    autoencoder_config = config["autoencoder_params"]
    train_config = config["train_params"]

    im_dataset_cls = {
        "mnist": MnistDataset,
        "celebhq": CelebDataset
    }.get(dataset_config["name"])
    
    im_dataset = im_dataset_cls(
        split="train",
        im_path=dataset_config["im_path"],
        im_size=dataset_config["im_size"],
        im_channels=dataset_config["im_channels"],
    )
    
    data_loader = DataLoader(
        im_dataset, batch_size=1, shuffle=False        
    )
    
    num_images = train_config["num_samples"]
    ngrid = train_config["num_grid_rows"]
    print("ngrid", ngrid)
    
    idxs = torch.randint(0, len(im_dataset)-1, (num_images,))
    ims = torch.cat([im_dataset[idx][None, :] for idx in idxs]).float()
    ims = ims.to(device)
    
    model = VQVAE(im_channels=dataset_config["im_channels"],
                  model_config=autoencoder_config).to(device)
    # model.load_state_dict(torch.load(os.path.join(train_config["task_name"], train_config["vqvae_autoencoder_ckpt_name"]), map_location=device))
    
    model.eval()
    
    with torch.no_grad():
        encoded_output, _ = model.encode(ims)
        decoded_output = model.decode(encoded_output)
        encoded_output = torch.clamp(encoded_output, -1., 1.)
        encoded_output = (encoded_output + 1) / 2
        decoded_output = torch.clamp(decoded_output, -1., 1.)
        decoded_output = (decoded_output + 1) / 2
        ims = (ims + 1) / 2
        
        input_grid = make_grid(ims.cpu(), nrow=ngrid)
        encoder_grid = make_grid(encoded_output.cpu(), nrow=ngrid)
        decoder_grid = make_grid(decoded_output.cpu(), nrow=ngrid)
        input_grid = torchvision.transforms.ToPILImage()(input_grid)
        encoder_grid = torchvision.transforms.ToPILImage()(encoder_grid)
        decoder_grid = torchvision.transforms.ToPILImage()(decoder_grid)
        
        input_grid.save(os.path.join(train_config["task_name"], "input_samples.png"))
        encoder_grid.save(os.path.join(train_config["task_name"], "encoder_samples.png"))
        decoder_grid.save(os.path.join(train_config["task_name"], "decoder_samples.png"))
        
        if train_config["save_latents"]:
            latent_path = os.path.join(train_config["task_name"], train_config["vqvae_latent_dir_name"])
            latent_fnames = glob.glob(os.path.join(train_config["task_name"], train_config["vqvae_latent_dir_name"], "*.pkl"))
            
            if not os.path.exists(latent_path):
                os.mkdir(latent_path)
                
            fname_latent_map = {}
            part_count = 0
            count = 0
            for idx, im in enumerate(tqdm(data_loader)):
                encoded_output, _ = model.encode(im.float().to(device))
                fname_latent_map[im_dataset.images[idx]] = encoded_output.cpu()
                
                if (count+1)%1000 == 0:
                    pickle.dump(fname_latent_map, open(os.path.join(latent_path, f"{part_count}.pkl"), "wb"))
                    part_count += 1
                    fname_latent_map = {}
            if len(fname_latent_map) > 0:
                pickle.dump(fname_latent_map, open(os.path.join(latent_path, f"{part_count}.pkl"), "wb"))
            
            print("Done Saving latents")
                
if __name__ == "__main__":
    parser = ArgumentParser(description="infer vqvae")
    parser.add_argument("--config", dest="config_path", default= "config/mnist.yaml", type=str)
    args = parser.parse_args()
    infer(args)
