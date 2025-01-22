import socket

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from diffusion import create_diffusion
from diffusion.gaussian_diffusion import ModelVarType

def get_gradient_vector(model: nn.Module):
    """
    Returns the current gradients of the model as rank-1 tensor of dim=#params.
    """
    return torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])


## TODO: Adjust for diffusion training (time-step dependence).
## TODO: Investigate EMA (Exp. Moving Avg) and Min-SNR(t) weighting
## TODO: Improve/Correct B_crit calculation.
## TODO: Refactor prints.
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
                 dataset,
                 model,
                 loss_fn,
                 betas: iter,
                 device: str,
                 data_portion=1.0,
                 verbose=True,
                 update=True
                 ):
        self.model = model.to(device)
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.optim = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0)
        self.betas = betas
        self.device = device
        self.verbose = verbose
        if verbose:
            print("\nGNS Initializing...")

        self.grad_log = []
        self.G_true = self.get_true_gradient(data_portion, update)
        print("G_true:",self.G_true)
        self.G2 = torch.norm(self.G_true) ** 2

        self.G_est = None  ## Current batch gradient
        self.snr = None    ## Current signal-to-noise ratio
        self.gns = None    ## Current gradient noise scale
        self.B_crit = self.critical_batch_size()
        print("B_crit:",self.B_crit)
        if verbose:
            print("\n---------GNS Initialized---------")
            print(f"Device: {device.upper()}")
            print(f"dim(G): {tuple(self.G_true.shape)}")
            print(f"G2: {float(self.G2):.5f}")
            print(f"GNS: {self.gns:.5f}")
            print(f"B_crit: {self.B_crit}")
            print("----------------------------------")

    def get_true_gradient(self, data_portion=1.0, update=True, verbose=True) -> Tensor:
        assert 0.0 < data_portion <= 1.0, "Data portion must be between 0 and 1."
        self.optim.zero_grad()
        self.model.train()
        grads: Tensor = ...
        
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
        diffusion.model_var_type = ModelVarType.LEARNED
        ## Downsample training data
        TOTAL_SIZE = int(len(self.dataset) * data_portion)
        BATCH_SIZE = min(32, TOTAL_SIZE)
        indices = np.random.permutation(len(self.dataset))[:TOTAL_SIZE]
        data = Subset(self.dataset, indices=indices)
        loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)
        #data = Subset(self.dataset, indices=np.random.randint(0, len(self.dataset), size=SIZE))
        #loader = DataLoader(data, batch_size=SIZE, shuffle=False)
        
        # Accumulate gradients
        accumulated_grads = None
        print("\n----------------------------------------------")
        print(f"\nCalculating G_true w.r.t {TOTAL_SIZE} data points:")
        
        for x, y in tqdm(loader, desc="Processing batches", disable=not verbose):
            x = x.to(self.device)
            y = y.to(self.device)
            t = torch.randint(0, 1000, (x.size(0),), device=self.device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            
            model_kwargs = dict(y=y)
            
            loss_dict = diffusion.training_losses(self.model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            loss.backward()
            
            current_grads = get_gradient_vector(self.model)
            
            if accumulated_grads is None:
                accumulated_grads = current_grads
            else:
                accumulated_grads += current_grads
                
            # Clear the gradients of the current batch
            self.optim.zero_grad()

        # Calculate the average gradient
        accumulated_grads /= len(loader)  ## fixme: not num_elems; count num_batches

        if update:
            # Manually set the gradient and update
            idx = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    numel = param.grad.numel()
                    param.grad.copy_(accumulated_grads[idx:idx + numel].view_as(param.grad))
                    idx += numel
            self.optim.step()
        
        self.model.eval()
        return accumulated_grads

    def gradient_SNR(self, G_est: Tensor) -> float:
        """
        Calculates the gradient noise scale equal to the sum of the variances of the individual gradient components,
        divided by the global norm of the gradient.
        Reference: An Empirical Model of Large Batch Training - Section 2.2
        """
        ## TODO: Checkout https://arxiv.org/pdf/2001.07384

        assert G_est.ndim == 1, "Gradient vector should be ndim=1"
        self.G_est = G_est
        self.grad_log.append(G_est)

        noise = torch.sum(torch.pow(self.G_true - G_est, 2))
        signal = self.G2
        self.snr = noise / signal

        return self.snr

    def gradient_noise_scale(self, B_big=200, B_small=20) -> float:
        """
        Calculates the 'unbiased' estimate of the simple noise scale
        Reference: An Empirical Model of Large Batch Training - Appendix A.1
        """
        
        ## (True) Batch-Gradients
        G_big = self.get_true_gradient(B_big / len(self.dataset), verbose=False)
        G_small = self.get_true_gradient(B_small / len(self.dataset), verbose=False)
        print("G_big:",G_big)
        print("G_small:",G_small)
        ## Unbiased |G_true|^2 estimate
        G2 = B_big * (torch.norm(G_big) ** 2) - B_small * (torch.norm(G_small) ** 2)
        G2 *= 1 / (B_big - B_small)

        ## Unbiased Cov(G_est) estimate
        S = torch.norm(G_small) ** 2 - torch.norm(G_big) ** 2
        S *= 1 / ((1 / B_small) - (1 / B_big))
        print("S:",S)
        ## Unbiased Gradient Noise Scale
        self.gns = S / G2

        return self.gns

    def critical_batch_size(self) -> int:
        """
        Critical Batch-Size computed as GNS.
        """
        ## TODO: correct implementation
        return abs(int(self.gradient_noise_scale()))

    def min_SNR(self, t:int, gamma=5) -> float:
        """
        Reference: Efficient Diffusion Training via Min-SNR Weighting Strategy
        """
        ## TODO: check implementation
        var_t = self.betas[t]
        snr_t = (var_t / np.sqrt(1-var_t)) ** 2
        w_loss = min(gamma, snr_t)
        self.G_est = w_loss * self.G_est

        return w_loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Host: {socket.gethostname()}")
    print(f"Device: {device.upper()}\n")
