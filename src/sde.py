from abc import ABC, abstractmethod

import pyro.distributions as dist
import torch
from pyro.distributions.distribution import Distribution
from torch import Tensor


class SDE(ABC):
    prior: Distribution

    def __init__(self, size, T=1.0, device=torch.device("cpu")):
        super().__init__()
        self.size = size
        self.T = T
        self.device = device

    @abstractmethod
    def diffusion(self, t, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def drift(self, t, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def marginal_prob(self, t, x: Tensor):
        ...

    @abstractmethod
    def get_sigma(self, t) -> Tensor:
        """
        Gets standard deviation of marginal prob as a scalar.
        """
        ...

    def reverse(self, score_fn):
        size = self.size
        T = self.T
        # prior = self.prior
        diffusion_fwd = self.diffusion
        drift_fwd = self.drift

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def drift(self, t, x: Tensor) -> Tensor:
                f = drift_fwd(t, x)
                g = diffusion_fwd(t, x)
                score = score_fn(t, x)
                return f - g**2 * score
                # return f - batch_mul(g**2, score)

            def diffusion(self, t, x: Tensor) -> Tensor:
                return diffusion_fwd(t, x)

        return RSDE(size=size, T=T)


class VESDE(SDE):
    def __init__(
        self, sigma_min=1e-3, sigma_max=50, size=1, T=1.0, device=torch.device("cpu")
    ):
        super().__init__(size, T, device)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.size = size
        self.prior = dist.MultivariateNormal(
            torch.zeros(self.size, device=device),
            self.sigma_max**2 * torch.eye(self.size, device=device),
        )

    def get_sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def marginal_prob(self, t, x):
        mean = x
        std = self.get_sigma(t) * torch.ones_like(x)
        return mean, std

    def diffusion(self, t, x):
        sigma = self.get_sigma(t)
        return (
            sigma
            * torch.sqrt(2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min)))
            * torch.ones_like(x)
        )

    def drift(self, t, x):
        return torch.zeros_like(x)


class VPSDE(SDE):
    def __init__(
        self, beta_min=0, beta_max=50, size=1, T=1.0, device=torch.device("cpu")
    ):
        super().__init__(size, T, device)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.size = size
        self.prior = dist.MultivariateNormal(
            torch.zeros(self.size, device=device), torch.eye(self.size, device=device)
        )

    def get_sigma(self, t):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        return torch.sqrt(1 - torch.exp(2.0 * log_mean_coeff))

    def marginal_prob(self, t, x):
        sigma = self.get_sigma(t)
        log_mean_coeff = 0.5 * (1 - sigma**2).log()
        mean = torch.exp(log_mean_coeff) * x
        std = sigma * torch.ones_like(x)
        return mean, std

    def diffusion(self, t, x):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        return torch.sqrt(beta_t)

    def drift(self, t, x):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        return -0.5 * beta_t * x
