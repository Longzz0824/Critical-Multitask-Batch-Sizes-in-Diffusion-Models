import socket
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def get_gradient_vector(model: nn.Module):
    """
    Returns the current gradients of the model as rank-1 tensor of dim=#params.
    """
    return torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])


class GradientNoiseScale:
    def __init__(self, model, dataset, loss_fn, device, verbose=True):
        self.model = model.to(device)
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.optim = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0)
        self.device = device

        self.grad_log = []
        self.G_true = self.get_true_gradient()
        self.G2 = torch.einsum("i,i ->", self.G_true, self.G_true)
        self.G_est = None  ## Current batch gradient

        if verbose:
            print("\n---------GNS Initialized---------")
            print(f"Device: {device.upper()}.")
            print(f"Grad shape: {self.G_true.shape}")
            print(f"G2: {float(self.G2)}")
            print("----------------------------------\n")

    def get_true_gradient(self, data_portion=1.0, update=True) -> Tensor:
        assert 0.0 < data_portion <= 1.0, "Data portion must be between 0 and 1."
        self.optim.zero_grad()
        self.model.train()

        ## Downsample training data
        SIZE = int(len(self.dataset) * data_portion)
        data = Subset(self.dataset, indices=np.random.randint(0, len(self.dataset), size=SIZE))
        loader = DataLoader(data, batch_size=SIZE, shuffle=False)
        grads: Tensor = ...
        print(f"Calculating G_true w.r.t {SIZE} data points")
        for x, _ in tqdm(loader):
            x = x.to(self.device)
            out = self.model(x)
            loss = self.loss_fn(out, x)
            loss.backward()
            grads = get_gradient_vector(self.model)
            if update:
                self.optim.step()
        print("-------------------------------------------")

        self.model.eval()
        return grads

    def signal_to_noise_ratio(self, G_est: Tensor) -> float:
        """
        Calculates the gradient noise scale equal to the sum of the variances of the individual gradient components,
        divided by the global norm of the gradient.
        """
        assert G_est.ndim == 1
        self.G_est = G_est
        self.grad_log.append(G_est)

        noise = torch.sum(torch.pow(self.G_true - G_est, 2))
        signal = self.G2

        return noise / signal

    def get_critical_batch_size(self):
        return  ## TODO



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Host: {socket.gethostname()}")
    print(f"Device: {device.upper()}\n")
