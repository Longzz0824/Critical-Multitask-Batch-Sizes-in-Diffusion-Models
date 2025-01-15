import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from GNS import GradientNoiseScale


def visualize_training_gns(GNS: GradientNoiseScale,
                           loss_log: iter,
                           gns_log: iter,
                           args,
                           figsize=(12, 8)):
    plt.figure(figsize=figsize)
    param = sum(p.numel() for p in GNS.model.parameters() if p.requires_grad)
    plt.suptitle(f"CIFAR10-Autoencoder Training (Batch={args.batch}, #Param={param})", fontsize=24)

    plt.subplot(2, 1, 1)
    plt.title("Training Loss", fontsize=20)
    plt.ylabel("L2 Loss", fontsize=16)
    plt.plot(loss_log)

    plt.subplot(2, 1, 2)
    plt.title("Gradient Noise Scale", fontsize=20)
    plt.ylabel("B_simple", fontsize=16)
    plt.xlabel("Training Iterations", fontsize=16)
    plt.plot(gns_log)

    plt.savefig(f"./visuals/{args.save_fig}.png")
    print(f"Figure saved at visuals/{args.save_fig}.png\n")
