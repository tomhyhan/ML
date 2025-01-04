import torch
import argparse
import yaml
from easydict import EasyDict
from linear_noise_scheduler import LinearNoiseScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    
    # SETTINGS
    with open(args.config_path) as yaml_file:
        try:
            config = EasyDict(yaml.safe_load(yaml_file))
        except yaml.YAMLError as e:
            print(e)
    
    diffusion_config = config.diffusion_params
    dataset_config = config.dataset_params
    diffusion_model_config = config.ldm_params
    autoencoder_model_config = config.autoencoder_params
    train_config = config.train_params
    
    print(diffusion_config)
    # Noise scheduler
    noise_scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config.num_timesteps,
        beta_start=diffusion_config.beta_start,
        beta_end=diffusion_config.beta_end
    )
    
    text_tokenizer = None
    text_model = None
    empty_text_embed = None
    condition_types = []
    condition_config = diffusion_model_config.condition_config
    print(condition_config)
    print(dataset_config['im_path'], dataset_config['im_size'], dataset_config['im_channels'])
    if condition_config is not None:
        condition_types = condition_config.condition_types
        if "text" in condition_types:
            # later
            pass
            
    # im_dataset_cls = {
    #     "mnist": Mn
    # }
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for training ldm")
    parser.add_argument("--config", dest="config_path", default="config/mnist.yaml", type=str)
    args= parser.parse_args()
    train(args)