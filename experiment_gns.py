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
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as T
from tqdm import tqdm

from diffusion import create_diffusion
from GNS import GradientNoiseScale
from utils import load_DiT_S2, FeatureDataset, set_seed_for_all


## TODO: Make gns calculation during training (gradient_snr)
def compute_gns_in_epoch():
    """
    Calculates gns values throughout one epoch along mini-batches.
    """
    ##
    pass


def initialize_gns(args: argparse.Namespace) -> GradientNoiseScale:
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
        B=args.B,
        b=args.b,
        reps=args.reps,
        data_portion=args.true_portion,
        accumulate=args.accumulate,
        verbose=args.verbose
    )
    return GNS


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    ## TODO: add helpers

    ## GNS parameters
    parser.add_argument("--model", type=str, default="./checkpoints/0750000.pt")
    parser.add_argument("--true_portion", "-p", type=float, default=0.1)
    parser.add_argument("--diff_steps", "-T", type=int, default=1000)
    parser.add_argument("--B", "-B", type=int, default=1_000)
    parser.add_argument("--b", "-b", type=int, default=100)
    parser.add_argument("--reps", "-r", type=int, default=10)
    parser.add_argument("--t_min", type=int, default=None)
    parser.add_argument("--t_max", type=int, default=None)

    ## Experiment options
    parser.add_argument("--accumulate", "-acc", action="store_true")
    parser.add_argument("--epoch", "-e", action="store_false")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no_seed", "-ns", action="store_true")
    parser.add_argument("--no_warnings", "-nw", action="store_false")

    ## Save options
    parser.add_argument("--csv_path", type=str, default="gns_log.csv")
    parser.add_argument("--save_fig", type=str, default="./visuals")

    ## Parse and print
    args = parser.parse_args()
    if args.verbose:
        print("---------Experiment Arguments---------\n")
        for arg, val in dict(args.__dict__).items():
            print(f"{arg}: {val}")
        print("--------------------------------------\n")

    return args


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nHost: {socket.gethostname()}")
    print(f"Device: {device.upper()}\n")

    args = parse_arguments()

    if args.no_warnings:
        ## For cleaner output
        warnings.filterwarnings("ignore")
        print("All warnings are disabled!\n")

    if not args.no_seed:
        ## For reproducability
        set_seed_for_all(42, device)

    ## Main computation
    gns = initialize_gns(args=args)

    if args.epoch:
        print("Calculating for an epoch: (WIP)\n")
        compute_gns_in_epoch()

    ## Save results/visuals
    ## TODO

    log = gns.gns_track
    ## pickle.dump(log, "experiment_gns_log.pkl")

    print("Done!\n")
