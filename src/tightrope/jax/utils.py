import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import Distribution


def net_to_dist(params, net, t, x_t) -> Distribution:
    """
    Wrap mean and log variance returned by a network in a batched `Normal`.
    """
    mean, log_var = net.apply(params, jnp.atleast_1d(t), jnp.atleast_2d(x_t))
    d = dist.Normal(mean, jnp.sqrt(jnp.exp(log_var)))  # type: ignore
    event_dims = x_t.ndim - 1
    return d.to_event(event_dims)


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)
