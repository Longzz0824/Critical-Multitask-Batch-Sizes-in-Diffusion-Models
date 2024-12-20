import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Union


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}\n")


class GradientNoiseScale:
    """
    For calculating diffusion timestep dependent largest useful Batch-Size.
    """
    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 loss_fn: nn.Module,
                 optimizer: optim.Optimizer = optim.SGD,
                 data_portion: float = 1.0,
                 simple=True,
                 device=device):

        ## Initial parameters
        self.model: nn.Module = model
        self.dataset, _ = random_split(dataset, lengths=(data_portion, 0))
        self.loss_fn: nn.Module = loss_fn
        self.optim = optimizer
        self.device = device
        self.simple = simple

        self.optim(self.model.parameters())  ## fixme

        ## GNS Variables
        self.G_true = self.get_true_gradient(data_portion)
        self.B_noise = ...
        self.e_opt = ...

        if self.simple:
            self.H = torch.eye(...)
        else:
            self.H = self.get_hessian()

    def __call__(self, batch: Tensor, t: int):
        """
        Calculates the Gradient Noise Scale given the batch. Updates true_grad if neccessary
        """
        ## return self.opt_batch_size()
        return batch.shape  ## TODO

    def get_batch_gradient(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Calculates the Gradient of the batch w.r.t. self.loss_fn
        """
        x, y = x.to(self.device), y.to(self.device)
        out = self.model(x)

        loss = self.loss_fn(out, y)
        loss.backward()
        self.optim.zero_grad()

        return x.grad

    def get_true_gradient(self, data_portion: float = 1.0) -> Tensor:
        """
        (True) Gradient of all/portion of dataset w.r.t. loss_fn
        """
        b_size = len(self.dataset)
        loader = DataLoader(self.dataset, batch_size=b_size, shuffle=False)
        for x, y in loader:
            return self.get_batch_gradient(x, y)

    def get_hessian(self) -> Tensor:
        """
        (True) Hessian of all/portion of dataset w.r.t. loss_fn
        """
        pass

    def gradient_noise_scale(self) -> float:
        ## TODO
        pass

    def opt_step_size(self, B: int) -> float:
        if self.simple:
            e_max = 1
        else:
            G = torch.einsum("ij, j -> i", self.H, self.G_true)
            e_max = torch.einsum("i,i->", self.G_true.T, G) / self.G_true.norm() ** 2

        return B * e_max / (B + self.B_noise)

    def opt_batch_size(self, simple=True) -> int:
        """
        Returns optimal batch size
        """
        pass
