from dataclasses import dataclass
from typing import Callable

import pyro.distributions as dist
import torch
from pyro.distributions import Distribution
from torch import Tensor


@dataclass
class NormalGFN:
    """
    Normal continuous GFlowNet for sampling proportional to the given reward function.

    Moves from `t=0` to `t=T` using a forward policy network :math:`F(x_{t+\\Delta t} | x_t)`.
    """

    log_R: Callable[[Tensor], Tensor]
    """Batched terminal log-reward function.
    """
    T: float
    n_steps: int
    x_0: Tensor
    """Initial state.
    """

    @property
    def dt(self):
        return self.T / self.n_steps

    def _get_ts_fwd(self, device):
        return torch.linspace(0, self.T - self.dt, self.n_steps, device=device)

    def _batch_x_0(self, n, device):
        """
        Get batched initial state.
        """
        return torch.repeat_interleave(self.x_0[None, :], n, dim=0).to(device)

    def _net_to_dist(self, net, t, x_t) -> Distribution:
        """
        Wrap mean and log variance returned by a network in a batched `Normal`.
        """
        mean, log_var = net(t, x_t)
        # Required for compatibility with with functorch.vmap
        d = dist.Normal(mean, log_var.exp().sqrt(), validate_args=False)
        # First dimension is for the batch
        event_dims = x_t.ndim - 1
        return d.to_event(event_dims)

    def sample(self, F_net, n):
        device = next(F_net.parameters()).device
        x_t = self._batch_x_0(n, device)
        ts = self._get_ts_fwd(device)

        for t in ts:
            t = t.repeat(n)

            # Sample x_{t+1} ~ F(x_{t+1} | x_t)
            F = self._net_to_dist(F_net, t, x_t)
            x_t = F.sample()

        return x_t

    def sample_trajectory(self, F_net, n):
        device = next(F_net.parameters()).device
        x_t = self._batch_x_0(n, device)
        ts = self._get_ts_fwd(device)

        x_ts = [x_t]

        for t in ts:
            t = t.repeat(n)

            # Sample x_{t+1} ~ F(x_{t+1} | x_t)
            F = self._net_to_dist(F_net, t, x_t)
            x_t = F.sample()

            x_ts.append(x_t)

        return torch.stack(x_ts)

    def get_loss_tb(self, F_net, B_net, log_Z_net, n):
        device = next(F_net.parameters()).device
        x_t = self._batch_x_0(n, device)
        ts = self._get_ts_fwd(device)
        loss = log_Z_net()

        for t in ts:
            t = t.repeat(n)

            # Sample x_{t+1} ~ F(x_{t+1} | x_t)
            F = self._net_to_dist(F_net, t, x_t)
            x_tp1 = F.sample()
            # ic(x_t, x_tp1)

            # Update loss
            tp1 = t + self.dt
            B = self._net_to_dist(B_net, tp1, x_tp1)
            loss = loss + F.log_prob(x_tp1) - B.log_prob(x_t)

            # Increment t
            x_t = x_tp1

        # Terminal reward
        loss = loss - self.log_R(x_t)

        return loss**2
