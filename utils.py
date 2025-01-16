import csv
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from GNS import GradientNoiseScale


def logger(args, csv_path="./train_log.csv"):
    args = vars(args)
    args["date"] = datetime.now().date()
    args["time"] = datetime.now().time()

    with open(csv_path, mode='a', newline="") as file:
        writer = csv.DictWriter(file, fieldnames=args.keys())
        writer.writeheader()
        writer.writerow(args)


def logger_to_dataframe(csv_path="./train_log.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)


def visualize_training_gns(GNS: GradientNoiseScale,
                           loss_log: iter,
                           gns_log: iter,
                           args,
                           figsize=(12, 8)):
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

