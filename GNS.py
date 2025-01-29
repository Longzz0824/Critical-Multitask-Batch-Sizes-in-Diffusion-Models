import socket

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from diffusion import SpacedDiffusion
from typing import Optional
from tqdm import tqdm


def get_gradient_vector(model: nn.Module) -> Tensor:
    """
    Returns the current gradients of the model as rank-1 tensor of dim=#params.
    """
    return torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])



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
                 data_portion: float = 0.2,
                 B: int = 1_000,
                 b: int = 100,
                 reps: int = 10,
                 t_min: Optional[int] = None,
                 t_max: Optional[int] = None,
                 initialize_gns=True,
                 accumulate=True,
                 verbose=True
                 ):
        ## Initial object variables
        self.model: nn.Module = model
        self.dataset: Dataset = dataset
        self.diff: SpacedDiffusion = diff
        self.optim = optim.AdamW(self.model.parameters())
        self.accumulate: bool = accumulate
        self.verbose: bool = verbose
        self.n_data = len(self.dataset)
        self.device: str = device

        ## Time interval for diffusion
        self.t_min: int = t_min
        self.t_max: int = t_max
        self.B: int = B
        self.b: int = b
        self.reps: int = reps
        
        ## Compute values
        self.gns_track: [float] = []
        self.gns: float = 0.
        self.b_true = int(self.n_data * data_portion)
        self.g_true: Tensor = self.get_true_gradient(data_portion)

        if initialize_gns:
            self.estimate_gns()

        if verbose:
            self.print_status()

    def print_status(self):
        print("---------GNS Initialized---------")
        print(f"Device: {self.device.upper()}")
        print(f"Grad Acc.: {self.accumulate}")
        print(f"Model: {self.model.__class__.__name__}\n")

        print(f"g_true: {tuple(self.g_true.shape)}")
        print(f"b_true: {int(self.b_true)}")
        print(f"gns: {self.gns:.4f}")
        print("----------------------------------\n")

    def get_random_batch(self, b_size: int, min_t=None, max_t=None):
        """
        Creates a batch of given size from the shuffled dataset. Optionally, one can
        limit the time steps in the batch.
        -----------------------------------------------------------------------------
        """
        ## Limiting time step interval (optional)
        min_t = min_t if min_t is not None else 0
        max_t = max_t if max_t is not None else self.diff.num_timesteps
        assert 0 <= min_t < max_t <= self.diff.num_timesteps, "Invalid time interval!"

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

        if self.accumulate:
            # Split into smaller sub-batches
            SUB_BATCH_SIZE = 32  # Adjust based on GPU memory
            n_samples = x.size(0)
            current_grads = None

            # Split data into sub-batches
            for i in range(0, n_samples, SUB_BATCH_SIZE):
                # Select sub-batch
                x_sub = x[i:i + SUB_BATCH_SIZE].to(self.device).squeeze(dim=1)
                y_sub = y[i:i + SUB_BATCH_SIZE].to(self.device).squeeze(dim=1)
                t_sub = t[i:i + SUB_BATCH_SIZE].to(self.device)

                # Prepare model input
                model_kwargs = dict(y=y_sub)

                # Calculate loss and backpropagate for the sub-batch
                loss_dict = self.diff.training_losses(self.model, x_sub, t_sub, model_kwargs)
                loss = loss_dict["loss"].mean()
                loss.backward()

                # Extract gradients for the sub-batch
                sub_batch_grads = get_gradient_vector(self.model)

                # Accumulate gradients
                if current_grads is None:
                    current_grads = sub_batch_grads
                else:
                    current_grads += sub_batch_grads

                # Clear gradients to free GPU memory
                self.optim.zero_grad()

            # Normalize gradients by the total number of sub-batches
            current_grads /= (n_samples // SUB_BATCH_SIZE + (n_samples % SUB_BATCH_SIZE != 0))

            self.model.eval()
            return current_grads
        
        else:
            ## Setup model input
            x = x.to(self.device).squeeze(dim=1)
            y = y.to(self.device).squeeze(dim=1)
            t = t.to(self.device)
            model_kwargs = dict(y=y)

            ## Calculate loss and backpropagate
            loss_dict = self.diff.training_losses(self.model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            loss.backward()

            ## Get gradients
            grads = get_gradient_vector(self.model)

            self.model.eval()
            return grads

    def get_true_gradient(self, data_portion: float) -> Tensor:
        """
        Calculates the expected gradient of the data set (or portion of it).
        The outcome will be treated as true update direction for the model.
        -----------------------------------------------------------------------------
        Reference: An Empirical Model of Large Batch Training - Section 2.2
        """
        assert 0. < data_portion <= 1., "Data portion must be between 0 and 1!\n"

        ## Create a random batch
        x, y, t = self.get_random_batch(self.b_true, min_t=self.t_min, max_t=self.t_max)

        if self.verbose:
            print(f"Calculating the true gradient (g_true) w.r.t {self.b_true} data points...\n")

        expected_grad = self.get_batch_gradient(x, y, t)

        return expected_grad

    def estimate_gns(self):
        """
        Estimates the 'unbiased' simple noise scale for larger datasets.
        -----------------------------------------------------------------------------
        Reference: An Empirical Model of Large Batch Training - Appendix A.1
        """
        assert self.B > self.b, "B can't be bigger than b!\n"
        assert self.reps > 1, "reps must be greater than 1\n!"

        B, b, reps = self.B, self.b, self.reps
        print(f"Estimating Unbiased GNS (B={B} b={b} reps={reps})")

        ## Estimate S: E[S] = E[|g_est - g_true|]^2
        small_batch = self.get_random_batch(b, min_t=self.t_min, max_t=self.t_max)
        big_batch = self.get_random_batch(B, min_t=self.t_min, max_t=self.t_max)
        B_grad = self.get_batch_gradient(*big_batch)
        b_grad = self.get_batch_gradient(*small_batch)

        S = (torch.norm(b_grad, p=2) ** 2 - torch.norm(B_grad, p=2) ** 2) / (1 / b - 1 / B)
        assert S > 0, "Noise became negative!\n"

        ## Estimate G2 over many (reps) batches: E[G2]^2 = E[g_true]^2
        G2s = []
        for _ in tqdm(range(reps)):
            small_batch = self.get_random_batch(b, min_t=self.t_min, max_t=self.t_max)
            big_batch = self.get_random_batch(B, min_t=self.t_min, max_t=self.t_max)
            B_grad = self.get_batch_gradient(*big_batch)
            b_grad = self.get_batch_gradient(*small_batch)

            G2 = (B * torch.norm(B_grad, p=2) ** 2 - b * torch.norm(b_grad, p=2)) / (B - b)
            if G2 < 0:
                continue

            G2s.append(G2)

        G2 = torch.mean(torch.stack(G2s), dim=0)
        assert G2 > 0, "Signal became negative!\n"

        ## Calculate unbiased gradient_snr
        self.gns = (S / G2).item()

        self.gns_track.append(self.gns)
        print(f"Estimation: {self.gns}\n")

    def gradient_snr(self, g_est: Tensor, b_size: int) -> float:
        """
        Calculates the gradient noise scale equal to the sum of the variances of the
        individual gradient components, divided by the global norm of the gradient.
        -----------------------------------------------------------------------------
        Reference: An Empirical Model of Large Batch Training - Section 2.2
        """
        assert g_est.shape == self.g_true.shape, "Gradient vectors must be in same shape!"
        assert b_size >= 1, "Batch size must be greater than 1!"

        gns = b_size * torch.norm((g_est - self.g_true), p=2) ** 2 / torch.norm(self.g_true, p=2) ** 2
        self.gns = gns
        self.gns_track.append(self.gns)

        return gns

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



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nHost: {socket.gethostname()}")
    print(f"Device: {device.upper()}\n")

