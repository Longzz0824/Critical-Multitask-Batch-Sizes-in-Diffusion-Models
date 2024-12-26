import socket
import numpy as np
import argparse

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
    ## Training initialization
    EPOCH = args.epoch
    B_SIZE = args.batch
    L_RATE = args.lr
    model = Autoencoder()
    model.to(device)
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=L_RATE, weight_decay=0)

    ## Data loading
    cifar10 = CIFAR10("./cifar10", train=True, transform=T.ToTensor(), download=True)
    loader = DataLoader(cifar10, batch_size=B_SIZE, shuffle=True)
    print(f"\nDataset: {len(cifar10)} * {tuple(cifar10[0][0].shape)}")
    print(f"Batch-Size: {B_SIZE}")

    ## Gradient Noise Scale
    GNS = GradientNoiseScale(
        model=model,
        dataset=cifar10,
        loss_fn=loss_fn,
        device=device
    )
    ## Training loop
    model.train()
    epoch_losses = []
    print("\nTraining started:\n")
    for epoch in range(EPOCH):
        losses = []
        for x, _ in tqdm(loader):
            opt.zero_grad()
            x = x.to(device)

            out = model(x)
            loss = loss_fn(out, x)
            losses.append(loss.item())

            ## Gradient Noise Scale
            G_est = get_gradient_vector(model)
            gns = GNS.signal_to_noise_ratio(G_est)
            print(f"GNS: {gns}")

            loss.backward()
            opt.step()

        with torch.no_grad():
            epoch_loss = np.mean(losses)
            epoch_losses.append(epoch_loss)
            print(f"[{epoch + 1}/{EPOCH}] Loss: {epoch_loss}")

    model.eval()
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=2, required=True)
    parser.add_argument("--batch", type=int, default=500, required=True)
    parser.add_argument("--lr", type=float, default=1e-4, required=True)
    parser.add_argument("--g_true", type=float, default=1.0, required=True)
    parser.add_argument("--save_fig", type=str, default="cifar_train")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Host: {socket.gethostname()}")
    print(f"Device: {device.upper()}")

    train_cifar(args, device=device)
