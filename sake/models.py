import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Union, List
from .layers import DenseSAKELayer

class DenseSAKEModel(nn.Module):
    hidden_features: int
    out_features: int
    depth: int = 4
    activation: Callable=nn.silu
    update: Union[List[bool], bool]=True

    def setup(self):
        self.embedding_in = nn.Dense(self.hidden_features)
        self.embedding_out = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.out_features),
            ],
        )

        if isinstance(self.update, bool):
            update = [self.update for _ in range(self.depth)]
        else:
            update = self.update

        for idx in range(self.depth):
            setattr(
                self,
                "d%s" % idx,
                DenseSAKELayer(
                    hidden_features=self.hidden_features,
                    out_features=self.hidden_features,
                    update=update[idx],
                ),
            )

        self.layers = [getattr(self, "d%s" % idx) for idx in range(self.depth)]

    def __call__(self, h, x, v=None, mask=None):
        h = self.embedding_in(h)
        for layer in self.layers:
            h, x, v = layer(h, x, v, mask=mask)
        h = self.embedding_out(h)
        return h, x, v


class RigidDocking(nn.Module):
    hidden_features: int
    out_features: int

    def setup(self):
        self.sake_model = DenseSAKEModel(self.hidden_features, self.out_features, update=False)
        self.fc_combine = nn.Sequential(
            (
                nn.Dense(self.sake_model.hidden_features),
                nn.silu,
            )
        )

        self.fc_origin = nn.Dense(1, use_bias=False)
        self.fc_x = nn.Dense(1, use_bias=False)
        self.fc_y = nn.Dense(1, use_bias=False)

    def match(self, h0, x0, h1, x1):
        h0 = self.sake_model.embedding_in(h0)
        h1 = self.sake_model.embedding_in(h1)

        for layer in self.sake_model.layers:
            h0, _, __ = layer(h0, x0)
            h1, _, __ = layer(h1, x1)
            h0_minus_h1 = jnp.expand_dims(h0, -2) - jnp.expand_dims(h1, -3)
            h0_dot_h1 = (jnp.expand_dims(h0, -2) * jnp.expand_dims(h1, -3)).sum(axis=-1, keepdims=True) \
                / h0.shape[-1] ** (0.5)

            a0 = jax.nn.softmax(h0_dot_h1, -2)
            a1 = jax.nn.softmax(h0_dot_h1, -3)

            m0 = (h0_minus_h1 * a0).sum(-2)
            m1 = (-h0_minus_h1 * a1).sum(-3)

            h0 = self.fc_combine(jnp.concatenate([h0, m0], -1))
            h1 = self.fc_combine(jnp.concatenate([h1, m1], -1))

        h0 = self.sake_model.embedding_out(h0)
        h1 = self.sake_model.embedding_out(h1)

        return h0, x0, h1, x1

    def anchor(self, h, x):
        origin, x_axis, y_axis = self.fc_origin(h), self.fc_x(h), self.fc_y(h)
        origin = origin * x
        x_axis = x_axis * x
        y_axis = y_axis * x
        x_axis = x_axis - origin
        y_axis = y_axis - origin
        return origin, x_axis, y_axis

    @staticmethod
    def change_of_coordinates_matrix(x_axis, y_axis):
        x_axis = x_axis / jnp.linalg.norm(x_axis, axis=-1, keepdims=True)
        y_axis = y_axis / jnp.linalg.norm(y_axis, axis=-1, keepdims=True)
        z_axis = jnp.cross(x_axis, y_axis)
        p_inv = jnp.stack([x_axis, y_axis, z_axis], -2)
        p = jnp.linalg.inv(p_inv)
        return p

    def transform(self, h, x):
        origin, x_axis, y_axis = self.anchor(h, x)
        p = self.change_of_coordinates_matrix(x_axis, y_axis)
        x = x - origin
        x = x @ p
        return x

    def __call__(self, h0, x0, h1, x1):
        h0, x0, h1, x1 = self.match(h0, x0, h1, x1)
        x0 = self.transform(h0, x0)
        x1 = self.transform(h1, x1)
        return x0, x1
