import csv
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T
from tqdm import tqdm

from GNS import GradientNoiseScale
from download import find_model
from models import DiT_models


F_DIR = "./features/imagenet256_features"
L_DIR = "./features/imagenet256_labels"
CSV_PATH = "./train_log.csv"
DATA_PATH = "./data"
device = "cuda" if torch.cuda.is_available() else "cpu"


## TODO: Improve if needed.
def one_epoch_gns(GNS: GradientNoiseScale, dataset: Dataset, b_size: int):
    """
    Calculates gns values throughout one epoch along mini-batches.
    """
    ## Testing gns calculation during training
    gns = 0
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True)
    for x, y in tqdm(dataloader):
        t = torch.randint(0, GNS.diff.num_timesteps, (b_size,))
        gns = GNS.get_batch_gradient(x, y, t)

    print(gns)
    return gns


def load_DiT_S2(path: str, device: str) -> nn.Module:
    model = DiT_models['DiT-S/2'](input_size=32, num_classes=1000).to(device)
    torch.serialization.add_safe_globals([path])
    state_dict = find_model(path)
    model.load_state_dict(state_dict)

    return model


## TODO: Complete implementation
def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)

    if device == "cpu":
        torch.manual_seed(seed)
    elif device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def simple_train_logger(args, csv_path=CSV_PATH):
    args = vars(args)
    args["date"] = datetime.now().date()
    args["time"] = datetime.now().time()

    with open(csv_path, mode='a', newline="") as file:
        writer = csv.DictWriter(file, fieldnames=args.keys())
        writer.writeheader()
        writer.writerow(args)


def logger_to_dataframe(csv_path=CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(csv_path)


## TODO: Adjust for GNS-Diffusion
def visualize_training_gns(GNS, loss_log: iter, gns_log: iter, args, figsize=(12, 8)):
    """
    Plots training loss and GNS of iterations,
    """
    plt.figure(figsize=figsize)
    param = sum(p.numel() for p in GNS.model.parameters() if p.requires_grad)
    plt.suptitle(f"CIFAR10-{args.model}AE Training (Batch={args.batch})", fontsize=24)

    plt.subplot(2, 1, 1)
    plt.title("Training Loss", fontsize=20)
    plt.ylabel("L2 Loss", fontsize=16)
    plt.plot(loss_log)

    plt.subplot(2, 1, 2)
    plt.title("Gradient Noise Scale", fontsize=20)
    plt.ylabel("B_simple", fontsize=16)
    plt.xlabel("Training Iterations", fontsize=16)
    plt.plot(gns_log)

    save_fig = f"{args.model}AE_e{args.epoch}_b{args.batch}"
    plt.savefig(f"./visuals/{save_fig}.png")
    print(f"Figure saved at visuals/{save_fig}.png\n")


def show_feature(img_no: int):
    dataset = ImageFolder(DATA_PATH, transform=T.Compose([T.Resize((256, 256)), T.ToTensor()]))
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    img, trgt = dataset[img_no]
    label = idx_to_class[trgt]
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"{label}", fontsize=18)

    ## Original image
    plt.subplot(1, 2, 1)
    plt.title(f"Original {tuple(img.shape)}", fontsize=12)
    plt.imshow(img.permute(1, 2, 0))
    plt.axis("off")

    ## Latent feature
    arr = np.load(f"{F_DIR}/{img_no}.npy")
    arr = np.clip(arr.squeeze(0), 0, 1)
    plt.subplot(1, 2, 2)
    plt.title(f"Feature {arr.shape}", fontsize=12)
    plt.imshow(arr.transpose(1, 2, 0))
    plt.axis("off")
    plt.show()


## TODO: Extend with transform arguments
class FeatureDataset(Dataset):
    def __init__(self, features_dir=F_DIR, labels_dir=L_DIR, transform=None):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.transform = transform

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))

        return torch.from_numpy(features), torch.from_numpy(labels)


class SmallAE(nn.Module):
    """
    Simple Autoencoder Network
    """

    def __init__(self, img_shape=(3, 32, 32)):
        super(SmallAE, self).__init__()
        C, H, W = img_shape
        self.device = device
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(C, H, W))
        self.encoder = nn.Sequential(
            nn.Linear(C * H * W, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, C * H * W)
        )

    def forward(self, x: Tensor):
        x = x.to(self.device)
        x = self.flatten(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.unflatten(x)
        return x


class LargeAE(nn.Module):
    def __init__(self, img_shape=(3, 32, 32)):
        super(LargeAE, self).__init__()
        C, H, W = img_shape
        self.device = device
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(C, H, W))
        self.encoder = nn.Sequential(
            nn.Linear(C * H * W, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, C * H * W)
        )

    def forward(self, x: Tensor):
        x = x.to(self.device)
        x = self.flatten(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.unflatten(x)
        return x
