"""
----------------------------------------------------
For non-diffusion gns experiments with ImageNet-256.
!!! Old and faulty version of GNS class !!!
----------------------------------------------------
"""
import argparse
import csv
import socket
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms as T
from tqdm import tqdm

from GNS import get_gradient_vector

CSV_PATH = "simple_train_log.csv"


class SimpleGNS:
    """
    Basic GNS Computations for Autoencoders.
    ------------------------------------------------------------------------------------
    References:
     + An Empirical Model of Large Batch Training (arXiv:1812.06162v1)
     + Efficient Diffusion Training via Min-SNR Weighting Strategy (arXiv:2303.09556v3)
     + Scalable Diffusion Models with Transformers (arXiv:2212.09748v2)
    ------------------------------------------------------------------------------------
    """

    def __init__(self, dataset, model, loss_fn, device, data_portion=1.0, verbose=True):
        self.model = model.to(device)
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.optim = optim.AdamW(self.model.parameters())
        self.device = device
        self.verbose = verbose
        if verbose:
            print("\nInitializing GNS...")

        self.G_true = self.get_true_gradient(data_portion)
        self.G2 = torch.einsum("i,i->", self.G_true, self.G_true)
        self.gns = 0  ## Current Gradient noise scale

        if verbose:
            print("\n---------GNS Initialized---------")
            print(f"Device: {device.upper()}")
            print(f"dim(G): {tuple(self.G_true.shape)}")
            print(f"G^2: {float(self.G2):.5f}")
            print(f"Initial GNS: {self.gns:.5f}")
            print("----------------------------------")

    def get_true_gradient(self, data_portion=1.0) -> Tensor:
        """
        Calculates the true gradient of the data set (or portion of it). The outcome will be treated as true
        update direction for the model.
        ----------------------------------------------------------------------------------------------------
        Reference: An Empirical Model of Large Batch Training - Section 2.2
        """
        assert 0.0 < data_portion <= 1.0, "Data portion must be between 0 and 1."
        self.optim.zero_grad()
        self.model.train()
        grads: Tensor = ...

        ## Downsample training data
        SIZE = int(len(self.dataset) * data_portion)
        data = Subset(self.dataset, indices=np.random.randint(0, len(self.dataset), size=SIZE))
        loader = DataLoader(data, batch_size=SIZE, shuffle=False)
        if self.verbose:
            print("\n----------------------------------------------")
            print(f"Calculating G_true w.r.t {SIZE} data points:")
        for x, _ in tqdm(loader, disable=not self.verbose):
            x = x.to(self.device)
            out = self.model(x)
            loss = self.loss_fn(out, x)
            loss.backward()
            grads = get_gradient_vector(self.model)

        self.model.eval()
        return grads

    def estimate_gns(self, B_big=30_000, B_small=1_000, reps=100):
        """
        Estimates the 'unbiased' simple noise scale for larger datasets.
        ----------------------------------------------------------------
        Reference: An Empirical Model of Large Batch Training - Appendix A.1
        """
        ## (True) Batch-Gradients
        G_big = self.get_true_gradient(B_big / len(self.dataset))
        G_small = self.get_true_gradient(B_small / len(self.dataset))

        ## Unbiased |G_true|^2 estimate (averaged)
        G2_s = []
        for _ in range(reps):
            G2 = B_big * (torch.norm(G_big, p=2) ** 2) - B_small * (torch.norm(G_small, p=2) ** 2)
            G2 *= 1 / (B_big - B_small)
            G2_s.append(G2)
        G2 = torch.mean(torch.stack(G2_s), dim=0)

        ## Unbiased Cov(G_est) estimate
        S = (torch.norm(G_small, p=2) ** 2) - (torch.norm(G_big, p=2) ** 2)
        S *= 1 / ((1 / B_small) - (1 / B_big))

        ## Unbiased Gradient Noise Scale
        self.gns = reps * (S / G2)

    def gradient_SNR(self, G_est: Tensor, b_size: int) -> float:
        """
        Calculates the gradient noise scale equal to the sum of the variances of the individual gradient components,
        divided by the global norm of the gradient.
        ------------------------------------------------------------------------------------------------------------
        Reference: An Empirical Model of Large Batch Training - Section 2.2
        """
        assert G_est.ndim == 1, "Gradient vector should be ndim=1"

        var = self.G_true - G_est
        noise = torch.einsum("i,i->", var, var)
        self.gns = b_size * (noise / self.G2)

        return self.gns

    def critical_batch_size(self, over_est=1) -> int:
        """
        Critical Batch-Size computed as GNS. Usually overestimates by a multiplicative factor.
        """
        return int(self.gns) // over_est

    def critical_l_rate(self) -> float:
        ## TODO: Check method
        lr_max = self.G2 / 2
        B = self.critical_batch_size()
        return float(lr_max / (1 + self.gns / B))


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


def simple_train_visualiser(GNS, loss_log: iter, gns_log: iter, args, figsize=(12, 8)):
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


def simple_train_logger(args, csv_path=CSV_PATH):
    args = vars(args)
    args["date"] = datetime.now().date()
    args["time"] = datetime.now().time()

    with open(csv_path, mode='a', newline="") as file:
        writer = csv.DictWriter(file, fieldnames=args.keys())
        writer.writeheader()
        writer.writerow(args)


def train_cifar(args, device):
    simple_train_logger(args)
    ## Arguments
    EPOCH = args.epoch
    B_SIZE = args.batch
    L_RATE = args.lr

    ## Data loading
    cifar10 = CIFAR10("./cifar10", train=True, transform=T.ToTensor(), download=True)
    loader = DataLoader(cifar10, batch_size=B_SIZE, shuffle=True)

    ## Training initialization
    if args.model == "small":
        model = SmallAE().to(device)
    elif args.model == "large":
        model = LargeAE().to(device)
    else:
        raise ValueError("Wrong model input")

    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=L_RATE, weight_decay=0)

    ## Gradient Noise Scale initialization
    GNS = SimpleGNS(
        model=model,
        dataset=cifar10,
        loss_fn=loss_fn,
        device=device,
        data_portion=args.g_true,
    )

    ## Training loop
    model.train()
    loss_track = []
    gns_track = []
    print("\n-------------Training Started-------------")
    for epoch in range(EPOCH):
        losses = []
        gns_scores = []
        for x, _ in tqdm(loader):
            opt.zero_grad()

            ## Backpropagation
            x = x.to(device)

            out = model(x)
            loss = loss_fn(out, x)
            loss.backward()

            ## Gradient Noise Scale calculation
            B = x.shape[0]
            G_est = get_gradient_vector(model)
            gns = GNS.gradient_SNR(G_est, b_size=B)

            ## Tracking
            gns_scores.append(float(gns))
            losses.append(loss.item())

            opt.step()

        with torch.no_grad():
            loss_track += losses
            gns_track += gns_scores

            epoch_loss = np.mean(losses)
            epoch_gns = np.mean(gns_scores)
            print(f"[{epoch + 1}/{EPOCH}] Loss: {epoch_loss:.4f}\tGNS: {epoch_gns:.4f}\t")

    model.eval()
    print("\n-------------Training Completed!--------------")

    return GNS, loss_track, gns_track


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=("small", "large"), default="large")
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--g_true", type=float, default=1.0)
    parser.add_argument("--B_big", type=int, default=30_000)
    parser.add_argument("--B_small", type=int, default=1_000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nHost: {socket.gethostname()}")
    print(f"Device: {device.upper()}")

    ## Training
    GNS, loss_log, gns_log = train_cifar(args, device=device)

    ## Visualiation
    simple_train_visualiser(GNS, loss_log, gns_log, args=args)
    print("Done!")
