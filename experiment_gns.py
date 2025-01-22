"""
------------------------------------------------------------------------------------------------------------------
Simple script for experimenting with gradient noise scale values in diffusion training. By default, the model is
DiT-S/2 and the dataset contains 50.000 compressed features (4, 32, 32) of the ImageNet-256 dataset.
------------------------------------------------------------------------------------------------------------------
"""
import os
import socket
import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision.transforms import transforms as T
from diffusion import create_diffusion
from GNS import GradientNoiseScale
from utils import load_DiT_S2, FeatureDataset



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nHost: {socket.gethostname()}")
    print(f"Device: {device.upper()}\n")

    ## Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/0750000.pt")
    parser.add_argument("--true_portion", type=float, default=0.2)
    parser.add_argument("--diff_steps", type=int, default=1000)
    parser.add_argument("--B", type=int, default=1_000)
    parser.add_argument("--b", type=int, default=100)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--t_min", type=int, default=None)
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--csv_path", type=str, default="gns_log.csv")
    parser.add_argument("--save_fig", type=str, default="./visuals")
    ## TODO: replace bool args with flags
    parser.add_argument("--init_gns", type=bool, default=True)  # needed?
    parser.add_argument("--accumulate", type=bool, default=True)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--augment", type=bool, default=True)
    args = parser.parse_args()

    ## Initialize model and diffusion
    model = load_DiT_S2(args.model_path, device=device)
    diff = create_diffusion("", diffusion_steps=args.diff_steps)

    ## Initialize dataset
    if args.augment:
        transform = T.Compose([
            ## TODO
        ])
    else: transform = T.ToTensor()

    features = FeatureDataset(transform=transform)  ## TODO: Complete  method in utils

    ## Initialize GNS module
    GNS = GradientNoiseScale(
        model=model,
        dataset=features,
        device=device,
        diff=diff,
        data_portion=args.true_portion,
        verbose=args.verbose
    )
    ## TODO: include t_min/t_max
    if args.init_gns:
        GNS = GNS.estimate_gns(B=args.B, b=args.b, reps=args.reps)


    # one_epoch_gns(GNS, features, 1000)

    ## Experiments...
    ## TODO

    ## Save results/visuals
    ## TODO
