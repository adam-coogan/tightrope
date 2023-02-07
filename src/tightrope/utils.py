import functorch
import pyro.distributions as dist
from pyro.distributions import Distribution


def net_to_dist(net, t, x_t) -> Distribution:
    """
    Wrap mean and log variance returned by a network in a batched `Normal`.
    """
    mean, log_var = net(t, x_t)
    # Required for compatibility with with functorch.vmap
    d = dist.Normal(mean, log_var.exp().sqrt(), validate_args=False)
    event_dims = x_t.ndim - 1
    return d.to_event(event_dims)


def batch_mul(a, b):
    return functorch.vmap(lambda a, b: a * b)(a, b)
