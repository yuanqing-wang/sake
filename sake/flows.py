import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from flax import linen as nn
from .models import DenseSAKEModel
from functools import partial
import math
from typing import Callable

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
        x = jax.random.normal(key=key, shape=shape)
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
    def dynamics_and_trace(model, params, key):
        dynamics = partial(ODEFlow.dynamics, model, params)
        trace = partial(ODEFlow.trace, dynamics)
        def fn(state, t):
            x, _trace = state
            return dynamics(x, t), trace(x, t, key)
        return fn

    @staticmethod
    def call(model, params, x, key):
        trace0 = jnp.zeros(shape=x.shape[:-2])
        fn = ODEFlow.dynamics_and_trace(model, params, key)
        y, logdet = odeint(fn, (x, trace0), T)
        y, logdet = y[-1], logdet[-1]
        return y, logdet

    @staticmethod
    def __call__(model, params, x, key): return ODEFlow.call(model, params, x, key)


class AugmentedFlowLayer(nn.Module):
    hidden_features: int=64
    depth: int=3
    activation: Callable=nn.silu
    def setup(self):
        import sake
        self.sake_model = sake.models.DenseSAKEModel(
            hidden_features=self.hidden_features,
            depth=self.depth,
            out_features=1,
            activation=self.activation,
        )
        self.scale_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(1, use_bias=False),
                jnp.tanh,
            ]
        )

    def mp(self, h, x):
        x0 = x
        h = jnp.concatenate([h, (x ** 2).sum(-1, keepdims=True)], axis=-1)
        h = jnp.concatenate([h, jnp.expand_dims(jnp.zeros_like(h[..., -1, :]), -2)], axis=-2)
        x = jnp.concatenate([x, jnp.expand_dims(jnp.zeros_like(x[..., -1, :]), -2)], axis=-2)
        h, x, _ = self.sake_model(h, x)
        x = x[..., :-1, :]
        h = h[..., :-1, :]
        translation = x - x0
        translation = translation - translation.mean(axis=-2, keepdims=True)
        scale = self.scale_mlp(h).mean(axis=-2, keepdims=True)
        return scale, translation

    def f_forward(self, h, x, v):
        scale, translation = self.mp(h, x)
        v = jnp.exp(scale) * v + translation
        log_det = scale.sum((-1, -2)) * v.shape[-1] * v.shape[-2]
        return x, v, log_det

    def f_backward(self, h, x, v):
        scale, translation = self.mp(h, x)
        v = v - translation
        v = jnp.exp(-scale) * v
        log_det = scale.sum((-1, -2)) * v.shape[-1] * v.shape[-2]
        return x, v, log_det

    def __call__(self, h, x, v): return self.f_forward(h, x, v)

class AugmentedFlowModel(nn.Module):
    depth: int=3
    mp_depth: int=3
    hidden_features: int=64
    activation: Callable=nn.silu
    def setup(self):
        for idx in range(self.depth):
            setattr(
                self,
                "xv_%s" % idx,
                AugmentedFlowLayer(self.hidden_features, self.mp_depth),
            )

            setattr(
                self,
                "vx_%s" % idx,
                AugmentedFlowLayer(self.hidden_features, self.mp_depth),
            )

        self.xv_layers = [getattr(self, "xv_%s" % idx) for idx in range(self.depth)]
        self.vx_layers = [getattr(self, "vx_%s" % idx) for idx in range(self.depth)]

    def f_forward(self, h, x, v):
        sum_log_det = 0.0
        for xv, vx in zip(self.xv_layers[::-1], self.vx_layers[::-1]):
            x, v, log_det = xv.f_forward(h, x, v)
            sum_log_det = sum_log_det + log_det

            v, x, log_det = vx.f_forward(h, v, x)
            sum_log_det = sum_log_det + log_det
        return x, v, sum_log_det

    def f_backward(self, h, x, v):
        sum_log_det = 0.0
        for xv, vx in zip(self.xv_layers, self.vx_layers):
            v, x, log_det = vx.f_backward(h, v, x)
            sum_log_det = sum_log_det + log_det

            x, v, log_det = xv.f_backward(h, x, v)
            sum_log_det = sum_log_det + log_det
        return x, v, sum_log_det

    def __call__(self, h, x, v): return self.f_forward(h, x, v)
