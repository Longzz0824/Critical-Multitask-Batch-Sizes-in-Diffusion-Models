import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Optional

device = "cuda" if torch.cuda.is_available() else "cpu"


class GradientNoiseScale:
    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 loss_fn: nn.Module,
                 device=device):

        self.model: nn.Module = model
        self.dataset: Dataset = dataset
        self.loss_fn: nn.Module = loss_fn
        self.optim = optim
        self.device = device
        self.n_data = len(dataset)


    def __call__(self, batch: Tensor):
        """
        Calculates the optimal batch size
        """
        ## return self.opt_batch_size()
        return batch.shape  ## TODO

    def batch_gradient(self, x: Tensor, y: Tensor) -> Tensor:
        """
        (Estimated) Gradient w.r.t. batch loss
        """
        x, y = x.to(self.device), y.to(self.device)
        out = self.model(x)
        loss = self.loss_fn(out, y)
        loss.backward()

        return x.grad

    def batch_variance(self, x: Tensor, y: Tensor) -> Tensor:
        pass

    def data_gradient(self) -> Tensor:
        """
        (True) Gradient w.r.t. loss over complete dataset
        """
        loader = DataLoader(self.dataset, self.n_data)
        for x, y in loader:
            return self.batch_gradient(x, y)

    def data_hessian(self) -> Tensor:
        pass

    def opt_step_size(self) -> float:
        pass

    def opt_batch_size(self, simple=True) -> int:
        """
        Returns optimal batch size
        """
        if simple:
            H = torch.eye(self.batch_size)
        else:
            H = self.data_hessian()

        ## TODO
        G_data = self.data_gradient()
        G2 = torch.einsum("", G_data, H, G_data)
        Var_data = ...

        return Var_data / G2
