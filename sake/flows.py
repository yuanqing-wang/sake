import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from flax import linen as nn
from .models import DenseSAKEModel
from functools import partial
import math

T = jnp.array((0.0, 1.0))

class CenteredGaussian(object):
    @staticmethod
    def log_prob(value):
        N = value.shape[-2]
        D = value.shape[-1]
        degrees_of_freedom = (N-1) * D
        r2 = jnp.reshape(value ** 2, (*value.shape[:-2], -1)).sum(-1)
        log_normalizing_constant = -0.5 * degrees_of_freedom * math.log(2*math.pi)
        log_px = -0.5 * r2 + log_normalizing_constant
        return log_px

    @staticmethod
    def sample(key, shape):
        x = jnp.random.normal(key=key, shape=shape)
        x = x - x.mean(axis=-2, keepdims=True)
        return x


class ODEFlow(object):
    @staticmethod
    def dynamics(model, params, x, t):
        t = jnp.ones((*x.shape[:-1], 1)) * t
        _, y, __ = model.apply(params, t, x)
        y = y - x
        return y

    @staticmethod
    def integrate(fn, x):
        return odeint(fn, x, T)[-1]

    @staticmethod
    def _jacobian(fn, x):
        jacobian = jax.jacrev(fn)(x)
        return jacobian

    @staticmethod
    def jacobian(fn, x):
        _jacobian = jax.vmap(partial(ODEFlow._jacobian, fn))
        return _jacobian(x)

    @staticmethod
    def trace(fn, x):
        res = fn(x)
        degrees_of_freedom = res.shape[-1] * res.shape[-2]
        res_shape = (*res.shape[:-4], degrees_of_freedom, degrees_of_freedom)
        res = jnp.reshape(res, res_shape)
        trace = jnp.trace(res, axis1=-2, axis2=-1)
        return trace

    @staticmethod
    def call(model, params, x):
        dynamics = partial(ODEFlow.dynamics, model, params)
        integrate = partial(ODEFlow.integrate, dynamics)
        jacobian = partial(ODEFlow.jacobian, integrate)
        trace = partial(ODEFlow.trace, jacobian)
        return integrate(x), trace(x)

    @staticmethod
    def __call__(model, params, x): return ODEFlow.call(model, params, x)
