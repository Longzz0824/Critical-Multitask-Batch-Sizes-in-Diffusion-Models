import socket
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from diffusion import SpacedDiffusion, create_diffusion
from download import find_model
from models import DiT_models


def get_gradient_vector(model: nn.Module) -> Tensor:
    """
    Returns the current gradients of the model as rank-1 tensor of dim=#params.
    """
    return torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])


## TODO: Adjust for diffusion training (time-step dependence).
class GradientNoiseScale:
    """
        Basic GNS Computations for Diffusion Training.
        ------------------------------------------------------------------------------------
        References:
         + An Empirical Model of Large Batch Training (arXiv:1812.06162v1)
         + Scalable Diffusion Models with Transformers (arXiv:2212.09748v2)
        ------------------------------------------------------------------------------------
        """

    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 diff: SpacedDiffusion,
                 device: str,
                 data_portion=0.2,
                 verbose=True
                 ):
        ## Object variables
        self.model: nn.Module = model
        self.dataset: Dataset = dataset
        self.diff: SpacedDiffusion = diff
        self.device: str = device
        self.verbose: bool = verbose

        self.optim = optim.AdamW(self.model.parameters())
        self.n_data: int = len(self.dataset)
        self.mem_size: float = (sys.getsizeof(self) / (1024 ** 3))

        ## Compute values
        self.gns_track: [float] = []
        self.gns: float = 0.
        self.b_true: int = 0
        self.g_true: Tensor = self.get_true_gradient(data_portion)

        if verbose:
            self.print_status()

    def print_status(self):
        print("\n---------GNS Initialized---------")
        print(f"Device: {self.device.upper()}")
        print(f"Memory: {self.mem_size:.2f} GiB")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"g_true: {tuple(self.g_true.shape)}")
        print(f"b_true: {float(self.b_true)}\n")
        print(f"Initial GNS: {self.gns:.4f}")
        print("----------------------------------")

    def get_random_batch(self, b_size: int, min_t=None, max_t=None):
        """
        Creates a batch of given size from the shuffled dataset. Optionally, one can
        limit the time steps in the batch.
        -----------------------------------------------------------------------------
        """
        ## Limiting time step interval (optional)
        min_t = min_t if min_t is not None else 0
        max_t = max_t if max_t is not None else self.diff.num_timesteps
        assert 0 <= min_t < max_t <= self.diff.num_timesteps

        ## TODO: Implement time step limits

        ## Return a random (shuffled) batch with additional t values
        for x, y in DataLoader(self.dataset, batch_size=b_size, shuffle=True, num_workers=0):
            t = torch.randint(min_t, max_t, size=(x.shape[0],))
            return x, y, t

    def get_batch_gradient(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        """
        Calculates the gradient vector of a given batch (x, y, t). Optionally, the
        gradients will be accumulated for GPU computation.
        -----------------------------------------------------------------------------
        """
        ## Setup model/optim
        self.model.train()
        self.optim.zero_grad()

        ## Setup model input
        x = x.to(self.device).squeeze(dim=1)
        y = y.to(self.device).squeeze(dim=1)
        t = t.to(self.device)
        model_kwargs = dict(y=y)

        ## Calculate loss and backpropagate
        loss_dict = self.diff.training_losses(self.model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        loss.backward()

        ## TODO: Gradient Accumulation

        ## Get gradients
        grads = get_gradient_vector(self.model)
        self.model.eval()

        return grads

    def get_true_gradient(self, data_portion):
        """
        Calculates the expected gradient of the data set (or portion of it).
        The outcome will be treated as true update direction for the model.
        -----------------------------------------------------------------------------
        Reference: An Empirical Model of Large Batch Training - Section 2.2
        """
        assert 0. < data_portion <= 1., "Data portion must be between 0 and 1."

        ## Create a random batch
        self.b_true = int(self.n_data * data_portion)
        x, y, t = self.get_random_batch(self.b_true)

        if self.verbose:
            print("\n----------------------------------------------")
            print(f"Calculating g_true w.r.t {self.b_true} data points:")

        return self.get_batch_gradient(x, y, t)

    def estimate_gns(self, B: int = 1_000, b: int = 100, reps=10):
        """
        Estimates the 'unbiased' simple noise scale for larger datasets.
        -----------------------------------------------------------------------------
        Reference: An Empirical Model of Large Batch Training - Appendix A.1
        """
        ## Estimate S: E[S] = E[|g_est - g_true|]^2
        small_batch = self.get_random_batch(b)
        big_batch = self.get_random_batch(B)
        B_grad = self.get_batch_gradient(*big_batch)
        b_grad = self.get_batch_gradient(*small_batch)

        S = (torch.norm(b_grad, p=2) ** 2 - torch.norm(B_grad, p=2) ** 2) / (1 / b - 1 / B)

        ## Estimate G2 over many (reps) batches: E[G2]^2 = E[g_true]^2
        G2s = []
        for _ in range(reps):
            small_batch = self.get_random_batch(b)
            big_batch = self.get_random_batch(B)
            B_grad = self.get_batch_gradient(*big_batch)
            b_grad = self.get_batch_gradient(*small_batch)

            G2 = (B * torch.norm(B_grad, p=2) ** 2 - b * torch.norm(b_grad, p=2)) / (B - b)
            G2s.append(G2)

        G2 = torch.mean(torch.stack(G2s), dim=0)

        ## Calculate unbiased gradient_snr
        self.gns = S / G2
        return self

    def gradient_snr(self, g_est: Tensor, b_size: int) -> float:
        """
        Calculates the gradient noise scale equal to the sum of the variances of the
        individual gradient components, divided by the global norm of the gradient.
        -----------------------------------------------------------------------------
        Reference: An Empirical Model of Large Batch Training - Section 2.2
        """
        assert g_est.shape == self.g_true.shape, "Gradient vectors must be in same shape."
        assert b_size >= 1, "Batch size must be greater than 1."
        return b_size * torch.norm((g_est - self.g_true), p=2) ** 2 / torch.norm(self.g_true, p=2) ** 2

    def critical_batch_size(self):
        """
        Critical Batch-Size computed as GNS. Usually overestimates by a multiplicative factor.
        """
        return int(self.gns)

    def critical_step_size(self, b_size: int):
        """
        Critical learning rate, optimal for the given batch size.
        """
        raise NotImplementedError


def test_GNS():
    from utils import FeatureDataset
    ## Initial variables
    B_SIZE = 64
    feature_dataset = FeatureDataset()
    diffusion = create_diffusion("")

    ## Load DiT-S/2 Model
    model = DiT_models['DiT-S/2'](input_size=32, num_classes=1000).to(device)
    PATH = "./checkpoints/0750000.pt"
    torch.serialization.add_safe_globals([PATH])
    state_dict = find_model(PATH)
    model.load_state_dict(state_dict)

    ## Testing GNS initialization
    GNS = GradientNoiseScale(
        model=model,
        dataset=FeatureDataset(),
        diff=diffusion,
        device=device
    ).estimate_gns(B=1_000, b=100, reps=10)

    ## Testing gns calculation during training
    gns = 0
    dataloader = DataLoader(feature_dataset, batch_size=B_SIZE, shuffle=True)
    for x, y in dataloader:
        t = torch.randint(0, diffusion.num_timesteps, (B_SIZE,))
        gns = GNS.get_batch_gradient(x, y, t)

    print(type(gns))
    print(gns)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nHost: {socket.gethostname()}")
    print(f"Device: {device.upper()}\n")

    test_GNS()
