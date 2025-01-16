import socket

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from diffusion import create_diffusion
from diffusion.respace import GaussianDiffusion

from torch import Tensor
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
from typing import Literal


def get_gradient_vector(model: nn.Module):
    """
    Returns the current gradients of the model as rank-1 tensor of dim=#params.
    """
    return torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])


## TODO: Adjust for diffusion training (time-step dependence).
## TODO: Investigate EMA (Exp. Moving Avg)
class GradientNoiseScale:
    """
    Basic GNS Computations for Diffusion Training.
    ------------------------------------------------------------------------------------
    References:
     + An Empirical Model of Large Batch Training (arXiv:1812.06162v1)
     + Efficient Diffusion Training via Min-SNR Weighting Strategy (arXiv:2303.09556v3)
     + Scalable Diffusion Models with Transformers (arXiv:2212.09748v2)
    ------------------------------------------------------------------------------------
    """
    def __init__(self,
                 dataset: Dataset,
                 model: nn.Module,
                 diffusion: GaussianDiffusion,
                 device: Literal["cpu", "cuda"],
                 data_portion = 1.0,
                 verbose=True,
                 ):
        self.model = model.to(device)
        self.dataset = dataset
        self.optim = optim.AdamW(self.model.parameters())
        self.diffusion = diffusion
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

        ## Calculating the gradient vector
        for x, y in tqdm(loader, disable=not self.verbose):
            x = x.to(self.device)
            y = y.to(self.device)
            t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
            model_kwargs = dict(y=y)
            loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            loss.backward()

            grads = get_gradient_vector(self.model)

        self.model.eval()
        return grads

    def estimate_gns(self, B_big=30_000, B_small=1_000, reps=100):
        """
        Estimates the 'unbiased' simple noise scale for larger datasets.
        ----------------------------------------------------------------
        Reference: An Empirical Model of Large Batch Training - Appendix A.1
        ## TODO: Check implementation (negative results)
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
        return float(lr_max / (1 + self.gns/B))



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Host: {socket.gethostname()}")
    print(f"Device: {device.upper()}\n")
