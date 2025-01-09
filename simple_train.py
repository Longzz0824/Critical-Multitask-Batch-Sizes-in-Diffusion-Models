import random
import socket
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms as T
from tqdm import tqdm

import diffusion
#######################################################
from GNS import GradientNoiseScale, get_gradient_vector
#######################################################


## TODO: check implementation
def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)

    if device == "cpu":
        torch.manual_seed(seed)
    elif device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class Autoencoder(nn.Module):
    """
    Simple Autoencoder Network
    """
    def __init__(self, img_shape=(3, 32, 32)):
        super(Autoencoder, self).__init__()
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


def train_cifar(args, device):
    ## Arguments
    EPOCH = args.epoch
    B_SIZE = args.batch
    L_RATE = args.lr

    ## Data loading
    cifar10 = CIFAR10("./cifar10", train=True, transform=T.ToTensor(), download=True)
    loader = DataLoader(cifar10, batch_size=B_SIZE, shuffle=True)

    ## Training initialization
    model = Autoencoder()
    model.to(device)
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=L_RATE, weight_decay=0)

    ## Gradient Noise Scale initialization
    GNS = GradientNoiseScale(
        model=model,
        dataset=cifar10,
        loss_fn=loss_fn,
        device=device,
        data_portion=args.g_true,
        betas=diffusion.create_diffusion("").betas
    )

    ## Training loop
    model.train()
    loss_log = []
    gns_log = []
    snr_log = []
    print("\n-------------Training Started-------------")
    for epoch in range(EPOCH):
        losses = []
        gns_scores = []
        snr_scores = []
        for x, _ in tqdm(loader):
            opt.zero_grad()
            x = x.to(device)
            out = model(x)
            loss = loss_fn(out, x)
            loss.backward()

            ## Gradient Noise Scale
            G_est = get_gradient_vector(model)
            gns = GNS.gradient_SNR(G_est)
            t = int(torch.randint(0, 1000, (1,), device=device))  # fixme
        #    snr = GNS.min_SNR(t, gamma=5)

            gns_scores.append(float(gns))
            losses.append(loss.item())
        #    snr_scores.append(float(snr))
            opt.step()

        with torch.no_grad():
            loss_log += losses
            gns_log += gns_scores
            snr_log += snr_scores

            epoch_loss = np.mean(losses)
            epoch_gns = np.mean(gns_scores)
        #    epoch_snr = np.mean(snr_scores)
            print(f"[{epoch + 1}/{EPOCH}] Loss: {epoch_loss}\tGNS: {epoch_gns}\tMin-SNR-t: {epoch_snr}")

    model.eval()
    print("\n-------------Training Completed!--------------")

    return GNS, loss_log, gns_log, snr_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--g_true", type=float, default=1.0)
    parser.add_argument("--save_fig", type=str, default="cifar_train")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nHost: {socket.gethostname()}")
    print(f"Device: {device.upper()}")

    GNS, loss_log, gns_log, snr_log = train_cifar(args, device=device)

    plt.figure(figsize=(16, 12))
    #g_size = int(len(GNS.dataset) * args.g_true)
    param = sum(p.numel() for p in GNS.model.parameters() if p.requires_grad)
    plt.suptitle(f"CIFAR10-Autoencoder Training (Batch={args.batch}, #Param={param})", fontsize=24)

    plt.subplot(3, 1, 1)
    plt.title("Training Loss", fontsize=20)
    plt.ylabel("L2 Loss", fontsize=16)
    plt.plot(loss_log)

    plt.subplot(3, 1, 2)
    plt.title("Gradient Noise Scale", fontsize=20)
    plt.ylabel("B_simple", fontsize=16)
    plt.plot(gns_log)

    plt.subplot(3, 1, 3)
    plt.title("Min-SNR(t)", fontsize=20)
    plt.xlabel("Optimization Step", fontsize=18)
    plt.ylabel("W_loss", fontsize=16)
    plt.plot(gns_log)

    plt.savefig(f"./visuals/{args.save_fig}.png")
    print(f"Figure saved at visuals/{args.save_fig}.png\n")
    print("Done!")
