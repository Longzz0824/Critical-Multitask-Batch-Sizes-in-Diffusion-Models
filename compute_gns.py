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
from datetime import datetime

import torch
from diffusion import create_diffusion
from GNS import GradientNoiseScale
from gns_utils import load_DiT_S2, FeatureDataset, set_seed_for_all, experiment_logger


CKPT_DIR = "checkpoints"
VISUAL_DIR = "visuals"
device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_gns_during_single_epoch():
    """
    Calculates gns values throughout one epoch along (large) mini-batches.
    """
    ## TODO: Make gns calculation during training (use GNS.gradient_snr() in training iterations)
    pass


def initialize_gns(args: argparse.Namespace) -> GradientNoiseScale:
    print("Initializing GNS object:\n")

    ## Initialize model and diffusion object
    model_ckpt = f"{args.ckpt_dir}/{args.model}"
    DiT = load_DiT_S2(model_ckpt, device=device)
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

    ## GNS parameters
    parser.add_argument("--model", type=str, choices=os.listdir(CKPT_DIR),
                        help="DiT-S/2 checkpoint")
    parser.add_argument("--true_portion", "-p", type=float, default=0.1,
                        help="Portion of dataset to compute g_true.")
    parser.add_argument("--diff_steps", "-T", type=int, default=1000,
                        help="Number of the diffusion time steps.")
    parser.add_argument("--B", "-B", type=int, default=500,
                        help="Big batch size for estimating gns.")
    parser.add_argument("--b", "-b", type=int, default=50,
                        help="Small batch size for estimating gns.")
    parser.add_argument("--reps", "-r", type=int, default=10,
                        help="Number of repetitions for estimating unbiased g_norm.")
    parser.add_argument("--t_min", type=int, default=None,
                        help="Floor value for diffusion steps.")
    parser.add_argument("--t_max", type=int, default=None,
                        help="Ceil value for diffusion steps.")
    ## Experiment options
    parser.add_argument("--accumulate", "-acc", action="store_true",
                        help="Whether to accumulate gradients when calling backward on batches.")
    parser.add_argument("--epoch", "-e", action="store_true",
                        help="Whether to compute gns values throughout single epoch.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Whether to print terminal.")
    parser.add_argument("--no_seed", "-ns", action="store_true",
                        help="Cancel seeds.")
    parser.add_argument("--no_warnings", "-nw", action="store_true",
                        help="Don't display warnings.")
    ## Save/load options
    parser.add_argument("--ckpt_dir", type=str, default=CKPT_DIR,
                        help="Directory where DiT-S/2 checkpoints exist.")
    parser.add_argument("--vis_dir", type=str, default=VISUAL_DIR,
                        help="Directory to save the figures into (if any).")
    parser.add_argument("--csv_path", type=str,
                        help="csv file for saving experiment results (log).")
    parser.add_argument("--save_fig", type=str, default=None,
                        help="png file name if there is any visual ouput.")

    ## Parse and print
    args = parser.parse_args()
    if args.verbose:
        print("---------Experiment Arguments---------\n")
        for arg, val in dict(args.__dict__).items():
            print(f"{arg}: {val}")
        print("--------------------------------------\n")

    return args


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.verbose:
        print(f"\nHost: {socket.gethostname()}")
        print(f"Device: {device.upper()}\n")

    if args.no_warnings:
        ## For cleaner output
        warnings.filterwarnings("ignore")
        print("All warnings are disabled!\n")

    if not args.no_seed:
        ## For reproducability
        set_seed_for_all(42, device)

    ## Main computation
    start_time = datetime.now()
    GNS = initialize_gns(args=args)
    gns_est = GNS.gns

    if args.epoch:
        print("Calculating for an epoch: (WIP)\n")
        compute_gns_during_single_epoch()

    ## Save the experiment
    experiment_logger(
        args,
        start=start_time,
        end=datetime.now(),
        gns_est=gns_est,
        g_norm=torch.norm(GNS.g_true, p=2),
        b_true=GNS.b_true
    )

    ## Confirm finish
    print("Done!")
    print(f"Results saved at: {args.csv_path}\n")



if __name__ == "__main__":
    args = parse_arguments()
    main(args)
