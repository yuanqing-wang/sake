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
    number_of_keypoints: int = 3

    def setup(self):
        self.sake_model = DenseSAKEModel(self.hidden_features, self.out_features, update=False)
        self.fc_combine = nn.Sequential(
            (
                nn.Dense(self.sake_model.hidden_features),
                nn.silu,
            )
        )

        self.fc_keypoints = nn.Dense(self.number_of_keypoints, use_bias=False)

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

    def keypoint(self, h0, x0, h1, x1):
        combination0 = self.fc_keypoints(h0)
        combination1 = self.fc_keypoints(h1)
        y0 = (jnp.expand_dims(x0, -2) * jnp.expand_dims(combination0, -1)).mean(-3)
        y1 = (jnp.expand_dims(x1, -2) * jnp.expand_dims(combination1, -1)).mean(-3)
        return y0, y1

    @staticmethod
    def kabsch(y1, y2):
        y1_bar = y1 - y1.mean(-2, keepdims=True)
        y2_bar = y2 - y2.mean(-2, keepdims=True)
        a = y2 @ y1.transpose()
        u2, s, u1 = jnp.linalg.svd(a)
        d = jnp.sign(jnp.det(u2 @ u1.transpose()))
        s = jnp.eye(3).at[2, 2].set(d)
        r = u2 @ s @ u1.transpose()
        t = y2.mean(-2, keepdims=True) - r * y1.mean(-2, keepdims=True)
        return r, t

    def __call__(self, h0, x0, h1, x1):
        h0, x0, h1, x1 = self.match(h0, x0, h1, x1)
        y0, y1 = self.keypoint(h0, x0, h1, x1)
        r, t = self.kabsch(y0, y1)
        y0 = y0 @ r + t
        return y0, y1

    
