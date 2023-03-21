from dataclasses import dataclass
from math import sqrt
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.numpy import ndarray as Array
from jax.random import split
from numpyro.distributions import Distribution

from .utils import net_to_dist


__all__ = ("NormalDiffPriorGFN",)


@dataclass
class NormalDiffPriorGFN:
    """
    Normal continuous GFlowNet for fitting posteriors where the prior is represented
    by a diffusion model.

    Moves from `t=T` to `t=0` using a backwards policy network :math:`\\tilde{B}(x_t | x_{t+\\Delta t})`.
    """

    # All must be batched
    get_f: Callable[[Array, Array], Array]
    get_g: Callable[[Array, Array], Array]
    get_score_prior_t: Callable[[Array, Array], Array]
    get_log_like: Callable[[Array], Array]
    T: float
    n_steps: int
    eps: float = 0.0
    """Minimum time value used to avoid numerical issues due to a vanishing diffusion
    coefficient.
    """

    @property
    def dt(self) -> float:
        return (self.T - self.eps) / self.n_steps

    @property
    def ts(self):
        # eps, ..., T
        return jnp.linspace(self.eps, self.T, self.n_steps + 1)

    def get_F(self, t, x_t) -> Distribution:
        """
        Prior forward (diffusion) process, p(x_{t+1} | x_t).
        """
        f_t = self.get_f(t, x_t)
        g_t = self.get_g(t, x_t)

        mean = x_t + f_t * self.dt
        std = sqrt(self.dt) * g_t * jnp.ones_like(x_t)
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
        std = sqrt(self.dt) * g_t * jnp.ones_like(x_t)
        d = dist.Normal(mean, std, validate_args=False)
        event_dims = x_t.ndim - 1
        return d.to_event(event_dims)

    def diffuse(self, x_0s):
        """
        Diffuse data from :math:`t=\\epsilson` to :math:`T`.
        """
        batch_size = x_0s.shape[0]

        x_t = x_0s
        ts = self.ts[:-1]  # eps, ..., T - dt
        for t in ts:
            t = t.repeat(batch_size)
            x_t = self.get_F(t, x_t).sample()

        return x_t

    def denoise(self, x_T):
        """
        Denoise prior samples from :math:`T` to :math:`t=\\epsilson`.
        """
        batch_size = x_T.shape[0]

        x_tp1 = x_T
        tp1s = self.ts[::-1][:-1]  # T, ..., eps + dt
        for tp1 in tp1s:
            tp1 = tp1.repeat(batch_size)
            x_tp1 = self.get_B(tp1, x_tp1).sample()

        return x_tp1

    def _sample_helper(self, key, params_Bt, Bt_net, x_T, save_traj=False) -> Array:
        """
        Generate posterior samples or full trajectories starting from high-temperature
        prior samples.
        """
        traj = [x_T] if save_traj else None
        x_tp1 = x_T
        tp1s = self.ts[::-1][:-1]  # T, ..., eps + dt
        for tp1 in tp1s:
            key, subkey = split(key)

            # Sample x_t ~ B(x_t | x_{t+1})
            x_tp1 = net_to_dist(params_Bt, Bt_net, tp1, x_tp1).sample(subkey)

            if save_traj:
                assert traj is not None
                traj.append(x_tp1)

        if save_traj:
            assert traj is not None
            return jnp.stack(traj)
        else:
            return x_tp1

    def sample(self, key, params_Bt, Bt_net, x_T):
        return self._sample_helper(key, params_Bt, Bt_net, x_T, False)

    def sample_trajectory(self, key, params_Bt, Bt_net, x_T):
        return self._sample_helper(key, params_Bt, Bt_net, x_T, True)

    def _sample_ais_helper(
        self,
        key,
        params_Ft,
        Ft_net,
        params_Bt,
        Bt_net,
        log_Z,
        x_T,
        save_traj=False,
    ) -> Tuple[Array, Array]:
        """
        Generate posterior samples or full trajectories starting from high-temperature
        prior samples along with (log) AIS weights.
        """
        traj = [x_T] if save_traj else None
        x_tp1s = x_T
        tp1s = self.ts[::-1][:-1]  # T, ..., eps + dt
        loss = log_Z
        for tp1 in tp1s:
            key, subkey = split(key)
            t = tp1 - self.dt

            # Sample x_t ~ B(x_t | x_{t+1})
            Bt = net_to_dist(params_Bt, Bt_net, tp1, x_tp1s)
            x_ts = Bt.sample(subkey)

            Ft = net_to_dist(params_Ft, Ft_net, t, x_ts)
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
            return jnp.stack(traj), -loss
        else:
            return x_tp1s, -loss

    def sample_ais(self, key, params_Ft, Ft_net, params_Bt, Bt_net, log_Z, x_T):
        return self._sample_ais_helper(
            key, params_Ft, Ft_net, params_Bt, Bt_net, log_Z, x_T, False
        )

    def sample_ais_trajectory(
        self, key, params_Ft, Ft_net, params_Bt, Bt_net, log_Z, x_T
    ):
        return self._sample_ais_helper(
            key, params_Ft, Ft_net, params_Bt, Bt_net, log_Z, x_T, True
        )

    def _get_loss_helper(self, key, params_Ft, Ft_net, params_Bt, Bt_net, log_Z, x_T):
        """
        Helper to sample the forward KL or on/off-policy TB losses.

        To-dos
            - Implement forward KL loss
            - Switch network between train and eval mode
            - Allow off-policy sampling
        """
        x_tp1s = x_T
        tp1s = self.ts[::-1][:-1]  # T, ..., eps + dt
        loss = log_Z
        for tp1 in tp1s:
            key, subkey = split(key)
            t = tp1 - self.dt

            # Sample x_t ~ B_θ(x_t | x_{t+1}), blocking the θ gradient
            x_ts = net_to_dist(
                jax.lax.stop_gradient(params_Bt), Bt_net, tp1, x_tp1s
            ).sample(subkey)

            Bt = net_to_dist(params_Bt, Bt_net, tp1, x_tp1s)
            Ft = net_to_dist(params_Ft, Ft_net, t, x_ts)
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

        return loss**2

    def get_loss_tb(
        self, key, params_Ft, Ft_net, params_Bt, Bt_net, log_Z, x_T: Array
    ):
        """
        Compute trajectory balance loss starting from high-temperature samples.
        """
        return self._get_loss_helper(
            key, params_Ft, Ft_net, params_Bt, Bt_net, log_Z, x_T
        )

    def get_loss_fwd_kl(self):
        """
        Compute forward KL loss starting from high-temperature samples.

        Seems to have a bug!
        """
        raise NotADirectoryError()
