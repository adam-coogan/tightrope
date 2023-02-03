from math import pi

import pyro.distributions as dist
import torch


def get_log_mu(x, y=0.4, sigma_x0=1.0, sigma_n=0.3):
    """
    Target distribution.
    """
    # Validating args breaks grad+vmap
    prior = dist.Normal(
        0.0, torch.full_like(x, sigma_x0), validate_args=False
    ).to_event(1)

    # Observation is nonlinear transformation of x with noise
    yp = torch.sin(pi * x)
    likelihood = dist.Normal(
        y, torch.full_like(x, sigma_n), validate_args=False
    ).to_event(1)

    return prior.log_prob(x) + likelihood.log_prob(yp)
