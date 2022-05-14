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
    def _jacobian(fn, t, x):
        jacobian = jax.jacrev(fn)(x, t)
        return jacobian

    @staticmethod
    def jacobian(fn, x, t):
        _jacobian = jax.vmap(partial(ODEFlow._jacobian, fn, t))
        return _jacobian(x)

    # @staticmethod
    # def trace(fn, x, t):
    #     res = fn(x, t)
    #     degrees_of_freedom = res.shape[-1] * res.shape[-2]
    #     res_shape = (*res.shape[:-4], degrees_of_freedom, degrees_of_freedom)
    #     res = jnp.reshape(res, res_shape)
    #     trace = jnp.trace(res, axis1=-2, axis2=-1)
    #     return trace

    @staticmethod
    def trace(fn, x, t, key):
        _fn = lambda x: fn(x, t)
        y, vjp_fun = jax.vjp(_fn, x)
        key, subkey = jax.random.split(key)
        u = jax.random.normal(subkey, y.shape)
        trace = vjp_fun(u)[0] * u
        trace = trace.sum(axis=(-1, -2))
        return trace

    @staticmethod
    def logdet(fn, x):
        res = fn(x)
        degrees_of_freedom = res.shape[-1] * res.shape[-2]
        res_shape = (*res.shape[:-4], degrees_of_freedom, degrees_of_freedom)
        res = jnp.reshape(res, res_shape)
        _, logdet = jnp.linalg.slogdet(res)
        return logdet

    @staticmethod
    def dynamics_and_trace(model, params):
        dynamics = partial(ODEFlow.dynamics, model, params)
        trace = partial(ODEFlow.trace, dynamics)
        def fn(state, t):
            x, _trace = state
            return dynamics(x, t), trace(x, t)
        return fn

    @staticmethod
    def call(model, params, x, key):
        trace0 = jnp.zeros(shape=x.shape[:-2])
        fn = ODEFlow.dynamics_and_trace(model, params)
        y, logdet = odeint(fn, (x, trace0), T)
        y, logdet = y[-1], logdet[-1]
        return y, logdet

    @staticmethod
    def __call__(model, params, x, key): return ODEFlow.call(model, params, x, key)
