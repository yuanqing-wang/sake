import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable
from .utils import ExpNormalSmearing
from .functional import get_x_minus_xt, get_x_minus_xt_norm, get_h_cat_ht
from functools import partial

class ContinuousFilterConvolutionWithConcatenation(nn.Module):
    out_features : int
    kernel_features : int = 50
    activation : Callable = jax.nn.silu

    def setup(self):
        self.kernel = ExpNormalSmearing()
        self.mlp_in = nn.Dense(self.kernel_features)
        self.mlp_out = nn.Sequential(
            [
                nn.Dense(self.out_features),
                self.activation,
                nn.Dense(self.out_features),
            ]
        )

    def __call__(self, h, x):
        h0 = h
        h = self.mlp_in(h)
        _x = self.kernel(x) * h

        h = self.mlp_out(
            jnp.concatenate(
                [h0, _x, x],
                axis=-1
            )
        )

        return h

class DenseSAKELayer(nn.Module):
    out_features : int
    hidden_features : int
    activation : Callable = jax.nn.silu
    n_heads : int = 4

    def setup(self):
        self.edge_model = ContinuousFilterConvolutionWithConcatenation(self.hidden_features)
        self.n_coefficients = self.n_heads * self.hidden_features

        self.node_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.out_features),
                self.activation,
            ]
        )

        self.velocity_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(1, use_bias=False,
                    kernel_init=nn.initializers.variance_scaling(
                        scale=0.001,
                        mode="fan_avg",
                        distribution="uniform",
                    ),
                ),
            ],
        )

        self.semantic_attention_mlp = nn.Sequential(
            [
                nn.Dense(self.n_heads),
                partial(nn.leaky_relu, negative_slope=0.2),
            ],
        )

        self.post_norm_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.hidden_features),
                self.activation,
            ]
        )

        self.v_mixing = nn.Dense(1, use_bias=False)

        log_gamma = -jnp.log(jnp.linspace(1.0, 5.0, 4))
        self.log_gamma = self.param(
            "log_gamma",
            nn.initializers.constant(log_gamma),
            log_gamma.shape,
        )

    def spatial_attention(self, h_e_mtx, x_minus_xt, x_minus_xt_norm, euclidean_attention, mask=None):
        # (batch_size, n, n, n_coefficients)
        # coefficients = self.coefficients_mlp(h_e_mtx)# .unsqueeze(-1)
        coefficients = h_e_mtx

        # (batch_size, n, n, 3)
        # x_minus_xt = x_minus_xt * euclidean_attention.mean(dim=-1, keepdim=True) / (x_minus_xt_norm + 1e-5)
        x_minus_xt = x_minus_xt / (x_minus_xt_norm + 1e-5) # ** 2

        # (batch_size, n, n, coefficients, 3)
        combinations = jnp.expand_dims(x_minus_xt, -2) * jnp.expand_dims(coefficients, -1)

        if mask is not None:
            combinations = combinations * jnp.expand_dims(jnp.expand_dims(mask, -1), -1)

        # (batch_size, n, n, coefficients)
        combinations_sum = combinations.mean(axis=-3)
        combinations_norm = (combinations_sum ** 2).sum(-1)# .pow(0.5)

        h_combinations = self.post_norm_mlp(combinations_norm)
        return h_combinations, combinations

    def aggregate(self, h_e_mtx, mask=None):
        # h_e_mtx = self.mask_self(h_e_mtx)
        if mask is not None:
            h_e_mtx = h_e_mtx * jnp.unsqueeze(mask, -1)
        h_e = h_e_mtx.sum(axis=-2)
        return h_e

    def node_model(self, h, h_e, h_combinations):
        out = jnp.concatenate([
                h,
                h_e,
                h_combinations,
            ],
            axis=-1)
        out = self.node_mlp(out)
        out = h + out
        return out

    def euclidean_attention(self, x_minus_xt_norm):
        # (batch_size, n, n, 1)
        _x_minus_xt_norm = x_minus_xt_norm + 1e5 * jnp.expand_dims(jnp.eye(
            x_minus_xt_norm.shape[-2],
            x_minus_xt_norm.shape[-2],
        ), -1)

        att = jax.nn.softmax(
            -_x_minus_xt_norm * jnp.exp(self.log_gamma),
            axis=-2,
        )
        return att

    def semantic_attention(self, h_e_mtx):
        # (batch_size, n, n, n_heads)
        att = self.semantic_attention_mlp(h_e_mtx)

        # (batch_size, n, n, n_heads)
        # att = att.view(*att.shape[:-1], self.n_heads)
        att = att - 1e5 * jnp.expand_dims(jnp.eye(
            att.shape[-2],
            att.shape[-2],
        ), -1)
        att = jax.nn.softmax(att, axis=-2)
        return att

    def combined_attention(self, x_minus_xt_norm, h_e_mtx):
        euclidean_attention = self.euclidean_attention(x_minus_xt_norm)
        semantic_attention = self.semantic_attention(h_e_mtx)
        combined_attention = jax.nn.softmax(euclidean_attention * semantic_attention, axis=-2)
        return euclidean_attention, semantic_attention, combined_attention

    def velocity_model(self, v, h):
        v = self.velocity_mlp(h) * v
        return v

    def __call__(
            self,
            h,
            x,
            v=None,
            mask=None,
        ):

        x_minus_xt = get_x_minus_xt(x)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt)
        h_cat_ht = get_h_cat_ht(h)

        h_e_mtx = self.edge_model(h_cat_ht, x_minus_xt_norm)

        euclidean_attention, semantic_attention, combined_attention = self.combined_attention(x_minus_xt_norm, h_e_mtx)
        h_e_att = jnp.expand_dims(h_e_mtx, -1) * jnp.expand_dims(combined_attention, -2)
        h_e_att = jnp.reshape(h_e_att, h_e_att.shape[:-2] + (-1, ))
        h_combinations, delta_v = self.spatial_attention(h_e_att, x_minus_xt, x_minus_xt_norm, combined_attention, mask=mask)
        delta_v = self.v_mixing(delta_v.swapaxes(-1, -2)).swapaxes(-1, -2).mean(axis=(-2, -3))

        # h_e_mtx = (h_e_mtx.unsqueeze(-1) * combined_attention.unsqueeze(-2)).flatten(-2, -1)
        h_e = self.aggregate(h_e_att, mask=mask)
        h = self.node_model(h, h_e, h_combinations)

        if v is not None:
            v = self.velocity_model(v, h)
        else:
            v = jnp.zeros_like(x)

        v = delta_v + v

        x = x + v


        return h, x, v