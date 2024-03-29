from dataclasses import dataclass
from math import sqrt
from typing import Any, Callable, Tuple

import numpyro.distributions as dist
from numpyro.distributions import Distribution

import jax
import jax.numpy as jnp
from jax.numpy import ndarray as Array
from jax.random import split

from .utils import net_to_dist

__all__ = ("NormalDiffPriorGFN", "make_gfn_apply")


@dataclass(frozen=True)
class NormalDiffPriorGFN:
    """
    Normal continuous GFlowNet for fitting posteriors where the prior is represented
    by a diffusion model.

    Moves from `t=T` to `t=0` using a backwards policy network :math:`\\tilde{B}(x_t | x_{t+\\Delta t})`.
    """

    sde: Any  # TODO: change to SDE
    get_score_prior_t: Callable[[Array, Array], Array]
    get_log_like: Callable[[Array], Array]
    n_steps: int
    eps: float = 0.0
    """Minimum time value used to avoid numerical issues due to a vanishing diffusion
    coefficient.
    """

    @property
    def dt(self) -> float:
        return (self.sde.T - self.eps) / self.n_steps

    @property
    def ts(self):
        # eps, ..., T
        return jnp.linspace(self.eps, self.sde.T, self.n_steps + 1)

    def get_F(self, t, x_t) -> Distribution:
        """
        Prior forward (diffusion) process, p(x_{t+1} | x_t).
        """
        f_t = self.sde.f(t, x_t)
        g_t = self.sde.g(t, x_t)

        mean = x_t + f_t * self.dt
        std = sqrt(self.dt) * g_t * jnp.ones_like(x_t)
        d = dist.Normal(mean, std, validate_args=False)  # type: ignore
        event_dims = x_t.ndim - 1
        return d.to_event(event_dims)

    def get_B(self, t, x_t) -> Distribution:
        """
        Prior backward (denoising) process, p(x_{t-1} | x_t).
        """
        f_t = self.sde.f(t, x_t)
        g_t = self.sde.g(t, x_t)
        score_prior_t = self.get_score_prior_t(t, x_t)

        # Note the minus sign on dt
        mean = x_t - (f_t - g_t**2 * score_prior_t) * self.dt
        std = sqrt(self.dt) * g_t * jnp.ones_like(x_t)
        d = dist.Normal(mean, std, validate_args=False)  # type: ignore
        event_dims = x_t.ndim - 1
        return d.to_event(event_dims)

    def diffuse(self, key, x_0):
        """
        Diffuse data from :math:`t=\\epsilson` to :math:`T`.
        """
        def body_fn(i, val):
            t = i * self.dt
            key, x_t = val
            key, subkey = split(key)
            x_t = self.get_F(t, x_t).sample(subkey)
            return key, x_t

        return jax.lax.fori_loop(0, self.n_steps, body_fn, (key, x_0))[1]

    # def denoise(self, x_T):
    #     """
    #     Denoise prior samples from :math:`T` to :math:`t=\\epsilson`.
    #     """
    #     batch_size = x_T.shape[0]

    #     x_tp1 = x_T
    #     tp1s = self.ts[::-1][:-1]  # T, ..., eps + dt
    #     for tp1 in tp1s:
    #         tp1 = tp1.repeat(batch_size)
    #         x_tp1 = self.get_B(tp1, x_tp1).sample()

    #     return x_tp1

    # def _sample_helper(self, key, params_Bt, apply_Bt, x_T, save_traj=False) -> Array:
    #     """
    #     Generate posterior samples or full trajectories starting from high-temperature
    #     prior samples.
    #     """
    #     traj = [x_T] if save_traj else None
    #     x_tp1 = x_T
    #     tp1s = self.ts[::-1][:-1]  # T, ..., eps + dt
    #     for tp1 in tp1s:
    #         key, subkey = split(key)

    #         # Sample x_t ~ B(x_t | x_{t+1})
    #         x_tp1 = net_to_dist(params_Bt, apply_Bt, tp1, x_tp1).sample(subkey)

    #         if save_traj:
    #             assert traj is not None
    #             traj.append(x_tp1)

    #     if save_traj:
    #         assert traj is not None
    #         return jnp.stack(traj)
    #     else:
    #         return x_tp1

    def sample(self, key, params_Bt, apply_Bt, x_T, **apply_Bt_kwargs):
        def body_fun(i, val):
            tp1 = jnp.atleast_1d(self.sde.T - i * self.dt)  # T, ..., eps + dt
            x_tp1, key = val
            key, subkey = split(key)
            x_t = net_to_dist(
                params_Bt, apply_Bt, tp1, x_tp1, **apply_Bt_kwargs
            ).sample(subkey)
            return x_t, key

        init_val = (jnp.atleast_2d(x_T), key)
        x_0, _ = jax.lax.fori_loop(0, self.n_steps, body_fun, init_val)
        return x_0

    # def sample_trajectory(self, key, params_Bt, apply_Bt, x_T):
    #     return self._sample_helper(key, params_Bt, apply_Bt, x_T, True)

    def _sample_ais_helper(
        self,
        key,
        params_Ft,
        apply_Ft,
        params_Bt,
        apply_Bt,
        log_Z,
        x_T,
        save_traj=False,
        apply_Ft_kwargs={},
        apply_Bt_kwargs={},
    ) -> Tuple[Array, Array]:
        """
        Generate posterior samples or full trajectories starting from high-temperature
        prior samples along with (log) AIS weights.

        TODO: update!
        """
        traj = [x_T] if save_traj else None
        x_tp1s = x_T
        tp1s = self.ts[::-1][:-1]  # T, ..., eps + dt
        loss = log_Z
        for tp1 in tp1s:
            key, subkey = split(key)
            t = tp1 - self.dt

            # Sample x_t ~ B(x_t | x_{t+1})
            Bt = net_to_dist(params_Bt, apply_Bt, tp1, x_tp1s, **apply_Bt_kwargs)
            x_ts = Bt.sample(subkey)

            Ft = net_to_dist(params_Ft, apply_Ft, t, x_ts, **apply_Ft_kwargs)
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

    def sample_ais(
        self,
        key,
        params_Ft,
        apply_Ft,
        params_Bt,
        apply_Bt,
        log_Z,
        x_T,
        apply_Ft_kwargs={},
        apply_Bt_kwargs={},
    ):
        return self._sample_ais_helper(
            key,
            params_Ft,
            apply_Ft,
            params_Bt,
            apply_Bt,
            log_Z,
            x_T,
            False,
            apply_Ft_kwargs,
            apply_Bt_kwargs,
        )

    def sample_ais_trajectory(
        self,
        key,
        params_Ft,
        apply_Ft,
        params_Bt,
        apply_Bt,
        log_Z,
        x_T,
        apply_Ft_kwargs={},
        apply_Bt_kwargs={},
    ):
        return self._sample_ais_helper(
            key,
            params_Ft,
            apply_Ft,
            params_Bt,
            apply_Bt,
            log_Z,
            x_T,
            True,
            apply_Ft_kwargs,
            apply_Bt_kwargs,
        )

    def _get_loss_helper(
        self,
        key,
        params_Ft,
        apply_Ft,
        params_Bt,
        apply_Bt,
        log_Z,
        x_T,
        block_sample_grad,
        apply_Ft_kwargs={},
        apply_Bt_kwargs={},
    ):
        """
        Helper to sample the forward KL or on/off-policy TB losses. Intended to
        be used with ``vmap``.

        To-dos
            - Implement forward KL loss
            - Switch network between train and eval mode
            - Allow off-policy sampling
            - Batch?
        """

        def body_fun(i, val):
            x_tp1, loss, key = val
            key, subkey = split(key)
            tp1 = jnp.atleast_1d(self.sde.T - i * self.dt)  # T, ..., eps + dt
            t = tp1 - self.dt  # T - dt, ..., eps

            # Sample x_t ~ B_θ(x_t | x_{t+1}), blocking the θ gradient
            x_t = net_to_dist(
                jax.lax.cond(
                    block_sample_grad, jax.lax.stop_gradient, lambda p: p, params_Bt
                ),
                apply_Bt,
                tp1,
                x_tp1,
                **apply_Bt_kwargs
            ).sample(subkey)

            # Update loss
            Bt = net_to_dist(params_Bt, apply_Bt, tp1, x_tp1, **apply_Bt_kwargs)
            Ft = net_to_dist(params_Ft, apply_Ft, t, x_t, **apply_Ft_kwargs)
            F = self.get_F(t, x_t)
            B = self.get_B(tp1, x_tp1)
            loss = (
                loss
                + (
                    Bt.log_prob(x_t)
                    - B.log_prob(x_t)
                    - Ft.log_prob(x_tp1)
                    + F.log_prob(x_tp1)
                ).sum()
            )

            return x_t, loss, key

        init_val = (jnp.atleast_2d(x_T), log_Z, key)
        x_0, loss, _ = jax.lax.fori_loop(0, self.n_steps, body_fun, init_val)
        loss = loss - self.get_log_like(x_0)
        return loss

    def get_loss_tb(
        self,
        key,
        params_Ft,
        apply_Ft,
        params_Bt,
        apply_Bt,
        log_Z,
        x_T: Array,
        block_sample_grad=True,
        apply_Ft_kwargs={},
        apply_Bt_kwargs={},
    ):
        """
        Compute trajectory balance loss starting from high-temperature samples.
        """
        return self._get_loss_helper(
            key,
            params_Ft,
            apply_Ft,
            params_Bt,
            apply_Bt,
            log_Z,
            x_T,
            block_sample_grad,
            apply_Ft_kwargs,
            apply_Bt_kwargs,
        ) ** 2

    def get_loss_fwd_kl(
        self,
        key,
        params_Ft,
        apply_Ft,
        params_Bt,
        apply_Bt,
        log_Z,
        x_T: Array,
        block_sample_grad=False,
        apply_Ft_kwargs={},
        apply_Bt_kwargs={},
    ):
        """
        Forward KL loss for training Bt.
        """
        return 2 * self._get_loss_helper(
            key,
            params_Ft,
            apply_Ft,
            params_Bt,
            apply_Bt,
            log_Z,
            x_T,
            block_sample_grad,
            apply_Ft_kwargs,
            apply_Bt_kwargs,
        )


def make_gfn_apply(apply_fn, get_f, get_g, dt, y):
    """
    Converts a network's ``apply`` method into a function evaluating the backward
    Euler-Maruyama step, which can then be passed to NormalDiffPriorGFN.

    Returns:
        Function taking ``variables``, ``t`` and ``x_t``.
    """

    def apply(variables, t, x_t, *args, **kwargs):
        f_t = get_f(t, x_t)
        g_t = get_g(t, x_t)
        score_t = apply_fn(variables, t, x_t, y[None], *args, **kwargs)
        mean = x_t - (f_t - g_t**2 * score_t) * dt
        std = g_t * jnp.sqrt(dt) * jnp.ones_like(x_t)
        return mean, jnp.log(std**2)

    return apply
