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
from utils import visualize_training_gns
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


## TODO: Integrate logger
def train_cifar(args, device):
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
    GNS = GradientNoiseScale(
        model=model,
        dataset=cifar10,
        loss_fn=loss_fn,
        device=device,
        data_portion=args.g_true,
        B_big=args.B_big,
        B_small=args.B_small,
        betas=diffusion.create_diffusion("").betas
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
            t = int(torch.randint(0, 1000, (1,), device=device))
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
            print(f"[{epoch+1}/{EPOCH}] Loss: {epoch_loss}\tGNS: {epoch_gns}\t")

    model.eval()
    print("\n-------------Training Completed!--------------")

    return GNS, loss_track, gns_track


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=("small", "large"), default="small")
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--g_true", type=float, default=1.0)
    parser.add_argument("--B_big", type=int, default=30_000)
    parser.add_argument("--B_small", type=int, default=1_000)
    parser.add_argument("--save_fig", type=str, default="cifar_train")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nHost: {socket.gethostname()}")
    print(f"Device: {device.upper()}")

    ## Training
    GNS, loss_log, gns_log = train_cifar(args, device=device)

    ## Visualiation
    visualize_training_gns(GNS, loss_log, gns_log, args=args)

    print("Done!")
