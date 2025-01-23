"""
------------------------------------------------------------------------------------------------------------------
Simple script for experimenting with gradient noise scale values in diffusion training. By default, the model is
DiT-S/2 and the dataset contains 50.000 compressed features (4, 32, 32) of the ImageNet-256 dataset.
------------------------------------------------------------------------------------------------------------------
"""
import os
import socket
import argparse
import warnings

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision.transforms import transforms as T
from diffusion import create_diffusion
from GNS import GradientNoiseScale
from utils import load_DiT_S2, FeatureDataset, set_seed_for_all


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="./checkpoints/0750000.pt")
    parser.add_argument("--true_portion", type=float, default=0.2)
    parser.add_argument("--B", type=int, default=1_000)
    parser.add_argument("--b", type=int, default=100)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--t_min", type=int, default=None)
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--diff_steps", type=int, default=1000)
    parser.add_argument("--csv_path", type=str, default="gns_log.csv")
    parser.add_argument("--save_fig", type=str, default="./visuals")
    ## Flags (bools)
    parser.add_argument("--accumulate", "-acc", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no_seed", "-ns", action="store_true")
    parser.add_argument("--no_warnings", "-nw", action="store_false")

    args = parser.parse_args()

    if args.verbose:
        print("\n---------Experiment Arguments---------\n")
        for arg, val in dict(args.__dict__).items():
            print(f"{arg}: {val}")
        print("--------------------------------------\n")

    return args


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nHost: {socket.gethostname()}")
    print(f"Device: {device.upper()}\n")

    ## Parse arguments
    args = parse_arguments()

    if args.no_warnings:
        warnings.filterwarnings("ignore")
        print("All warnings are disabled !\n")

    if not args.no_seed:
        set_seed_for_all(42, device)

    ## Initialize model and diffusion object
    DiT = load_DiT_S2(args.model, device=device)
    diff = create_diffusion("", diffusion_steps=args.diff_steps)
    print("DiT-S/2 model initialized succesfully.\n")

    ## Initialize dataset
    features = FeatureDataset()
    print(f"Feature-Dataset loaded: {len(features)} * {tuple(features[0][0].shape)}\n")

    ## Initialize GNS module
    GNS = GradientNoiseScale(
        model=DiT,
        dataset=features,
        device=device,
        diff=diff,
        t_min=args.t_min,
        t_max=args.t_max,
        data_portion=args.true_portion,
        accumulate=args.accumulate,
        verbose=args.verbose
    ).estimate_gns(B=args.B, b=args.b, reps=args.reps)

    ## one_epoch_gns(GNS, features, 1000)

    ## Experiments...
    ## TODO

    ## Save results/visuals
    ## TODO

    print("Done!\n")
