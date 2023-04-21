import numpyro.distributions as dist
from numpyro.distributions import Distribution

import jax
import jax.numpy as jnp


def net_to_dist(params, apply_fn, t, x_t, *apply_args, **apply_kwargs) -> Distribution:
    """
    Wrap mean and log variance returned by a network in a batched `Normal`.

    Args:
        t: shape (b,)
        x_t: shape (b, ...)

    Returns:
        Distribution with event shape ... and batch shape b
    """
    mean, log_var = apply_fn({"params": params}, t, x_t, *apply_args, **apply_kwargs)
    # Remove batch dim
    d = dist.Normal(mean, jnp.sqrt(jnp.exp(log_var)))  # type: ignore
    event_dims = x_t.ndim - 1
    return d.to_event(event_dims)


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)
