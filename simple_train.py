import socket
import argparse
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

#######################################################
from GNS import GradientNoiseScale, get_gradient_vector
#######################################################


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
        device=device
    )

    ## Training loop
    model.train()
    loss_log = []
    gns_log = []
    print("\nTraining started:")
    for epoch in range(EPOCH):
        losses = []
        gns_scores = []
        for x, _ in tqdm(loader):
            opt.zero_grad()
            x = x.to(device)
            out = model(x)
            loss = loss_fn(out, x)
            loss.backward()

            ## Gradient Noise Scale
            G_est = get_gradient_vector(model)
            gns = GNS.signal_to_noise_ratio(G_est)

            gns_scores.append(float(gns))
            losses.append(loss.item())
            opt.step()

        with torch.no_grad():
            loss_log += losses
            gns_log += gns_scores

            epoch_loss = np.mean(losses)
            epoch_gns = np.mean(gns_scores)
            print(f"[{epoch + 1}/{EPOCH}] Loss: {epoch_loss}\tGNS: {epoch_gns}")

    model.eval()
    print("\n----------Training completed!----------")

    return loss_log, gns_log


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

    loss_log, gns_log = train_cifar(args, device=device)

    plt.figure(figsize=(16, 12))
    plt.suptitle(f"CIFAR10 AE-Training (B={args.batch})", fontsize=24)

    plt.subplot(2, 1, 1)
    plt.title("Epoch Loss", fontsize=20)
    plt.ylabel("Loss", fontsize=16)
    plt.plot(loss_log)

    plt.subplot(2, 1, 2)
    plt.title("Gradient Noise Scale", fontsize=20)
    plt.xlabel("Iteration", fontsize=18)
    plt.ylabel("B_simple", fontsize=16)
    plt.plot(gns_log)

    plt.savefig(f"./visuals/{args.save_fig}.png")
    print(f"Figure saved at visuals/{args.save_fig}.png\n")
