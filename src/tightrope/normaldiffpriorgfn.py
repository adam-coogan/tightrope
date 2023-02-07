from dataclasses import dataclass
from math import sqrt
from typing import Callable, Optional, Tuple

import pyro.distributions as dist
import torch
from pyro.distributions import Distribution
from torch import Tensor
from torch.nn import Module

from .utils import batch_mul, net_to_dist


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
        std = sqrt(self.dt) * batch_mul(g_t, torch.ones_like(x_t))
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
        mean = x_t - (f_t - batch_mul(g_t**2, score_prior_t)) * self.dt
        std = sqrt(self.dt) * batch_mul(g_t, torch.ones_like(x_t))
        d = dist.Normal(mean, std, validate_args=False)
        event_dims = x_t.ndim - 1
        return d.to_event(event_dims)

    def diffuse(self, x_0s):
        """
        Diffuse data.
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
        Denoise prior samples.
        """
        batch_size = x_Ts.shape[0]

        x_tp1 = x_Ts
        tp1s = self._get_tp1s_denoise(x_Ts.device)
        for tp1 in tp1s:
            tp1 = tp1.repeat(batch_size)
            x_tp1 = self.get_B(tp1, x_tp1).sample()

        return x_tp1

    def _sample_helper(self, Bt_net, x_Ts, save_traj=False) -> Tensor:
        """
        Generate posterior samples or full trajectories starting from high-temperature
        prior samples.
        """
        batch_size = x_Ts.shape[0]
        device = x_Ts.device

        traj = [x_Ts] if save_traj else None
        x_tp1 = x_Ts
        tp1s = self._get_tp1s_denoise(device)
        for tp1 in tp1s:
            tp1 = tp1.repeat(batch_size)

            # Sample x_t ~ B(x_t | x_{t+1})
            Bt = net_to_dist(Bt_net, tp1, x_tp1)
            x_tp1 = Bt.sample()

            if save_traj:
                assert traj is not None
                traj.append(x_tp1)

        if save_traj:
            assert traj is not None
            return torch.stack(traj)
        else:
            return x_tp1

    def sample(self, Bt_net, x_Ts):
        return self._sample_helper(Bt_net, x_Ts, False)

    def sample_trajectory(self, Bt_net, x_Ts):
        return self._sample_helper(Bt_net, x_Ts, True)

    def _sample_ais_helper(
        self, Ft_net, Bt_net, log_Z_net, x_Ts, save_traj=False
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate posterior samples or full trajectories starting from high-temperature
        prior samples along with (log) AIS weights.
        """
        batch_size = x_Ts.shape[0]
        device = x_Ts.device

        traj = [x_Ts] if save_traj else None
        x_tp1s = x_Ts
        tp1s = self._get_tp1s_denoise(device)
        loss = log_Z_net()
        for tp1 in tp1s:
            tp1 = tp1.repeat(batch_size)
            t = tp1 - self.dt

            # Sample x_t ~ B(x_t | x_{t+1})
            Bt = net_to_dist(Bt_net, tp1, x_tp1s)
            x_ts = Bt.sample()

            Ft = net_to_dist(Ft_net, t, x_ts)
            F = self.get_F(t, x_ts)
            B = self.get_B(tp1, x_tp1s)
            loss = loss + (
                Bt.log_prob(x_ts)
                - B.log_prob(x_ts)
                - Ft.log_prob(x_tp1s)
                + F.log_prob(x_tp1s)
            )

            # Increment
            x_tp1s = x_ts

            if save_traj:
                assert traj is not None
                traj.append(x_tp1s)

        # Terminal reward
        loss = loss - self.get_log_like(x_tp1s)

        if save_traj:
            assert traj is not None
            return torch.stack(traj), -loss
        else:
            return x_tp1s, -loss

    def sample_ais(self, Ft_net, Bt_net, log_Z_net, x_Ts):
        return self._sample_ais_helper(Ft_net, Bt_net, log_Z_net, x_Ts, False)

    def sample_ais_trajectory(self, Ft_net, Bt_net, log_Z_net, x_Ts):
        return self._sample_ais_helper(Ft_net, Bt_net, log_Z_net, x_Ts, True)

    def _get_loss_helper(self, tb, Ft_net, Bt_net, log_Z_net, x_Ts, train_sampler=None):
        """
        Helper to sample the forward KL or on/off-policy TB losses.
        """
        batch_size = x_Ts.shape[0]
        device = x_Ts.device

        x_tp1s = x_Ts
        tp1s = self._get_tp1s_denoise(device)
        loss = log_Z_net()
        for tp1 in tp1s:
            tp1 = tp1.repeat(batch_size)
            t = tp1 - self.dt

            # Sample x_t ~ B(x_t | x_{t+1})
            if tb:
                training = Bt_net.training
                Bt_net.eval()
                with torch.no_grad():
                    if train_sampler is None:
                        x_ts = net_to_dist(Bt_net, tp1, x_tp1s).sample()
                    else:
                        x_ts = train_sampler(tp1, x_tp1s)

                if training:
                    Bt_net.train()
                Bt = net_to_dist(Bt_net, tp1, x_tp1s)
            else:
                Bt = net_to_dist(Bt_net, tp1, x_tp1s)
                x_ts = Bt.sample()

            Ft = net_to_dist(Ft_net, t, x_ts)
            F = self.get_F(t, x_ts)
            B = self.get_B(tp1, x_tp1s)
            loss = loss + (
                Bt.log_prob(x_ts)
                - B.log_prob(x_ts)
                - Ft.log_prob(x_tp1s)
                + F.log_prob(x_tp1s)
            )

            # Increment
            x_tp1s = x_ts

        # Terminal reward
        loss = loss - self.get_log_like(x_tp1s)

        if tb:
            return loss**2
        else:
            return 2 * loss

    def get_loss_tb(
        self,
        Ft_net: Module,
        Bt_net: Module,
        log_Z_net: Module,
        x_Ts: Tensor,
        train_sampler: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ):
        """
        Compute trajectory balance loss starting from high-temperature samples.
        """
        return self._get_loss_helper(
            True, Ft_net, Bt_net, log_Z_net, x_Ts, train_sampler
        )

    def get_loss_fwd_kl(
        self, Ft_net: Module, Bt_net: Module, log_Z_net: Module, x_Ts: Tensor
    ):
        """
        Compute forward KL loss starting from high-temperature samples.

        Seems to have a bug!
        """
        return self._get_loss_helper(False, Ft_net, Bt_net, log_Z_net, x_Ts)
