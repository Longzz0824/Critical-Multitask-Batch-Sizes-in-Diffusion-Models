import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T
from tqdm import tqdm

from GNS import GradientNoiseScale
from download import find_model
from models import DiT_models

F_DIR = "./features/imagenet256_features"
L_DIR = "./features/imagenet256_labels"
DATA_PATH = "./data"


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

        features = torch.from_numpy(features)
        labels = torch.from_numpy(labels)

        if self.transform is not None:
            features = self.transform(features)
            labels = self.transform(labels)

        return features, labels


def one_epoch_gns(GNS: GradientNoiseScale, dataset: Dataset, b_size: int):
    """
    Calculates gns values throughout one epoch along mini-batches.
    TODO: Improve if needed.
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


def set_seed_for_all(seed: int, device: str):
    ## For Python and Numpy
    random.seed(seed)
    np.random.seed(seed)

    ## For PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device == "cpu":
        torch.manual_seed(seed)
    elif device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"Python, Numpy, PyTorch seeds({seed}) are set in {device.upper()}.\n")


def logger_to_dataframe(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


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
