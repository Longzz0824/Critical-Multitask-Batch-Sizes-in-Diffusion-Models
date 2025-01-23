import os
import sys
import pwd
import random
import csv
import socket
from argparse import Namespace
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T


F_DIR = "./features/imagenet256_features"
L_DIR = "./features/imagenet256_labels"
DATA_DIR = "./data"


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


def load_DiT_S2(path: str, device: str) -> nn.Module:
    from download import find_model
    from models import DiT_models

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

    print(f"Python, Numpy, PyTorch seeds ({seed}) are set in {device.upper()}.\n")


def show_feature(img_no: int):
    dataset = ImageFolder(DATA_DIR, transform=T.Compose([T.Resize((256, 256)), T.ToTensor()]))
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


def experiment_logger(args: Namespace, gns_est: float, start: datetime, end: datetime):
    ## Handle arguments
    args = dict(args.__dict__)
    path = args.pop("csv_path")

    args["date"] = str(start.replace(microsecond=0))
    args["gns_est"] = gns_est
    args["runtime"] = str(end - start)
    args["user"] = pwd.getpwuid(os.getuid())[0]
    args["host"] = socket.gethostname()

    if os.path.exists(path):
        with open(path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(args.values())
    else:
        with open(path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(args.keys())
            writer.writerow(args.values())

    print(f"Experiment saved at {path}.\n")


def log_to_dataframe(path, raw=False):
    assert os.path.exists(path), f"Couldn't find file: {path}\n"
    df = pd.read_csv(path)

    ## Adjust data types
    df["date"] = pd.to_datetime(df["date"])
    df["runtime"] = pd.to_datetime(df["runtime"])
    df["runtime"] = df["runtime"].dt.time

    if raw:
        return df

    ## Dataframes
    df_meta = df[
        "date user host ckpt_dir vis_dir save_fig accumulate epoch verbose no_seed no_warnings runtime".split()
    ]
    df_param = df[
        "model diff_steps true_portion t_min t_max".split()
    ]
    df_result = df["gns_est runtime t_min t_max".split()
    ]
    return df_meta, df_param, df_result


def create_experiment_bash_with(args: str, bash_file="run_experiments.sh"):
    exp_script = f"python experiment_gns.py {args}\n"
    if os.path.exists(bash_file):
        with open(bash_file, "a") as file:
            file.write(exp_script)
    else:
        content = f"""#!/bin/bash\n"""
        with open(bash_file, "w") as file:
            file.write(content)
            file.write(exp_script)

        print(f"Created {bash_file} !\n")
