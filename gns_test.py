import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from gns_1 import GradientNoiseScale, get_gradient_vector

from train import CustomDataset
from diffusers.models import AutoencoderKL
from models import DiT_S_2
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T
from diffusion import create_diffusion

from typing import Optional, List   


def load_checkpoint(checkpoint_filename):
    image_size = 256 #@param [256, 512]
    vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
    latent_size = int(image_size) // 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiT_S_2(input_size=latent_size).to(device)
    checkpoints_dir = "./checkpoints"
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_filename)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    #print("Checkpoint keys:", checkpoint.keys())
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
features_dir = "./features/imagenet256_features"
labels_dir = "./features/imagenet256_labels"
dataset = CustomDataset(features_dir, labels_dir)

model = load_checkpoint("0010000.pt")

#use diffusion.training_losses instead of loss_fn here
loss_fn = nn.MSELoss()

betas = np.linspace(0.1, 0.2, num=1000)
device = "cuda" if torch.cuda.is_available() else "cpu"
gns = GradientNoiseScale(
    dataset=dataset,
    model=model,
    loss_fn=loss_fn,
    betas=betas,
    device=device,
    data_portion = 0.01,  
    verbose=True
)
print("critical_batch_size: ", gns.B_crit)
a = gns.gradient_noise_scale(1000, 50)
print("-------------------------------------")
print("GNS",a)

