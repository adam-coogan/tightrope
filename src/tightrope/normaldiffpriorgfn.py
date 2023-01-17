from dataclasses import dataclass
from math import sqrt
from typing import Callable

import pyro.distributions as dist
import torch
from pyro.distributions import Distribution
from torch import Tensor

from .utils import net_to_dist


@dataclass
class NormalDiffPriorGFN:
    """
    Normal continuous GFlowNet for fitting posteriors where the prior is represented
    by a diffusion model.

    Moves from `t=T` to `t=0` using a backwards policy network :math:`\\tilde{B}(x_t | x_{t+\\Delta t})`.
    """

    # All must be batched
    get_f: Callable[[Tensor, Tensor], Tensor]
    get_g: Callable[[Tensor, Tensor], Tensor]
    get_score_prior_t: Callable[[Tensor, Tensor], Tensor]
    get_log_like: Callable[[Tensor], Tensor]
    T: float
    n_steps: int

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    def _get_ts_diffuse(self, device):
        # [0, ..., T - dt]
        return torch.linspace(0, self.T - self.dt, self.n_steps, device=device)

    def _get_tp1s_denoise(self, device):
        # [T, ..., dt]
        return torch.linspace(self.T, self.dt, self.n_steps, device=device)

    def get_F(self, t, x_t) -> Distribution:
        """
        Prior forward (diffusion) process, p(x_{t+1} | x_t).
        """
        f_t = self.get_f(t, x_t)
        g_t = self.get_g(t, x_t)

        mean = x_t + f_t * self.dt
        std = g_t * sqrt(self.dt) * torch.ones_like(x_t)
        d = dist.Normal(mean, std, validate_args=False)
        event_dims = x_t.ndim - 1
        return d.to_event(event_dims)

    def get_B(self, t, x_t) -> Distribution:
        """
        Prior backward (denoising) process, p(x_{t-1} | x_t).
        """
        f_t = self.get_f(t, x_t)
        g_t = self.get_g(t, x_t)
        score_prior_t = self.get_score_prior_t(t, x_t)

        # Note the minus sign on dt
        mean = x_t - (f_t - g_t**2 * score_prior_t) * self.dt
        std = g_t * sqrt(self.dt) * torch.ones_like(x_t)
        d = dist.Normal(mean, std, validate_args=False)
        event_dims = x_t.ndim - 1
        return d.to_event(event_dims)

    def diffuse(self, x_0s):
        """
        Diffuse some data.
        """
        batch_size = x_0s.shape[0]

        x_t = x_0s
        ts = self._get_ts_diffuse(x_0s.device)
        for t in ts:
            t = t.repeat(batch_size)
            x_t = self.get_F(t, x_t).sample()

        return x_t

    def denoise(self, x_Ts):
        """
        Denoise some prior samples.
        """
        batch_size = x_Ts.shape[0]

        x_tp1 = x_Ts
        tp1s = self._get_tp1s_denoise(x_Ts.device)
        for tp1 in tp1s:
            tp1 = tp1.repeat(batch_size)
            x_tp1 = self.get_B(tp1, x_tp1).sample()

        return x_tp1

    def sample(self, Bt_net, x_Ts):
        """
        Generate posterior samples from high-temperature prior samples.
        """
        batch_size = x_Ts.shape[0]
        device = x_Ts.device

        x_tp1 = x_Ts
        tp1s = self._get_tp1s_denoise(device)
        for tp1 in tp1s:
            tp1 = tp1.repeat(batch_size)

            # Sample x_t ~ B(x_t | x_{t+1})
            Bt = net_to_dist(Bt_net, tp1, x_tp1)
            x_tp1 = Bt.sample()

        return x_tp1

    def sample_trajectory(self, Bt_net, x_Ts):
        """
        Generate trajectories for posterior samples starting from high-temperature
        prior samples.
        """
        batch_size = x_Ts.shape[0]
        device = x_Ts.device

        x_tp1 = x_Ts
        x_tp1s = [x_tp1]
        tp1s = self._get_tp1s_denoise(device)
        for tp1 in tp1s:
            tp1 = tp1.repeat(batch_size)

            # Sample x_t ~ B(x_t | x_{t+1})
            Bt = net_to_dist(Bt_net, tp1, x_tp1)
            x_tp1 = Bt.sample()

            x_tp1s.append(x_tp1)

        return torch.stack(x_tp1s)

    def _get_loss_helper(self, tb, Ft_net, Bt_net, log_Z_net, x_Ts):
        batch_size = x_Ts.shape[0]
        device = x_Ts.device

        x_tp1 = x_Ts
        tp1s = self._get_tp1s_denoise(device)
        # The evidence p(y)
        loss = log_Z_net()
        for tp1 in tp1s:
            tp1 = tp1.repeat(batch_size)
            t = tp1 - self.dt

            # Sample x_t ~ B(x_t | x_{t+1})
            if tb:
                # Sample on policy; don't differentiate through sampling
                Bt_net.eval()
                with torch.no_grad():
                    Bt_sampler = net_to_dist(Bt_net, tp1, x_tp1)
                    x_t = Bt_sampler.sample()
                Bt_net.train()
                Bt = net_to_dist(Bt_net, tp1, x_tp1)
            else:
                Bt = net_to_dist(Bt_net, tp1, x_tp1)
                x_t = Bt.sample()

            # Update loss
            Ft = net_to_dist(Ft_net, t, x_t)
            F = self.get_F(t, x_t)
            B = self.get_B(tp1, x_tp1)
            loss = loss + Bt.log_prob(x_t) - B.log_prob(x_t)
            loss = loss - (Ft.log_prob(x_tp1) - F.log_prob(x_tp1))

            # Increment
            x_tp1 = x_t

        # Terminal reward
        loss = loss - self.get_log_like(x_tp1)

        if tb:
            return loss**2
        else:
            return 2 * loss

    def get_loss_tb(self, Ft_net, Bt_net, log_Z_net, x_Ts):
        """
        Compute trajectory balance loss starting from some high-temperature samples.
        """
        return self._get_loss_helper(True, Ft_net, Bt_net, log_Z_net, x_Ts)

    def get_loss_fwd_kl(self, Ft_net, Bt_net, log_Z_net, x_Ts):
        """
        Compute forward KL loss starting from some high-temperature samples.

        Seems to have a bug!
        """
        return self._get_loss_helper(False, Ft_net, Bt_net, log_Z_net, x_Ts)
