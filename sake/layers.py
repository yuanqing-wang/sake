import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Optional
from .utils import ExpNormalSmearing, segment_mean, segment_softmax
from .functional import (
    get_x_minus_xt, get_x_minus_xt_norm, get_h_cat_ht,
    get_x_minus_xt_sparse, get_h_cat_ht_sparse,
)
from functools import partial
def double_sigmoid(x):
    return 2.0 * jax.nn.sigmoid(x)

class ContinuousFilterConvolutionWithConcatenation(nn.Module):
    out_features : int
    kernel_features : int = 50
    activation : Callable = jax.nn.silu

    def setup(self):
        self.kernel = ExpNormalSmearing(num_rbf=self.kernel_features)
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

class SAKELayer(nn.Module):
    out_features : int
    hidden_features : int
    activation : Callable = jax.nn.silu
    n_heads : int = 4
    update: bool=True
    use_semantic_attention: bool = True
    use_euclidean_attention: bool = True
    use_spatial_attention: bool = True
    cutoff: Callable = None


    def setup(self):
        self.edge_model = ContinuousFilterConvolutionWithConcatenation(self.hidden_features)
        self.n_coefficients = self.n_heads * self.hidden_features

        self.node_mlp = nn.Sequential(
            [
                # nn.LayerNorm(),
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.out_features),
                self.activation,
            ]
        )

        if self.update:
            self.velocity_mlp = nn.Sequential(
                [
                    nn.Dense(self.hidden_features),
                    self.activation,
                    nn.Dense(1, use_bias=False),
                    double_sigmoid,
                ],
            )

        self.semantic_attention_mlp = nn.Sequential(
            [
                nn.Dense(self.n_heads),
                partial(nn.celu, alpha=2.0),
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
        self.x_mixing = nn.Sequential([nn.Dense(self.n_coefficients, use_bias=False), jnp.tanh])

        log_gamma = -jnp.log(jnp.linspace(1.0, 5.0, self.n_heads))
        if self.use_semantic_attention and self.use_euclidean_attention:
            self.log_gamma = self.param(
                "log_gamma",
                nn.initializers.constant(log_gamma),
                log_gamma.shape,
            )
        else:
            self.log_gamma = jnp.ones(self.n_heads)

class DenseSAKELayer(SAKELayer):
    def spatial_attention(self, h_e_mtx, x_minus_xt, x_minus_xt_norm, mask=None):
        # (batch_size, n, n, n_coefficients)
        # coefficients = self.coefficients_mlp(h_e_mtx)# .unsqueeze(-1)
        coefficients = self.x_mixing(h_e_mtx)

        # (batch_size, n, n, 3)
        # x_minus_xt = x_minus_xt * euclidean_attention.mean(dim=-1, keepdim=True) / (x_minus_xt_norm + 1e-5)
        x_minus_xt = x_minus_xt / (x_minus_xt_norm + 1e-5) # ** 2

        # (batch_size, n, n, coefficients, 3)
        combinations = jnp.expand_dims(x_minus_xt, -2) * jnp.expand_dims(coefficients, -1)

        if mask is not None:
            _mask = jnp.expand_dims(jnp.expand_dims(mask, -1), -1)
            combinations = combinations * _mask
            combinations_sum = combinations.sum(axis=-3) / (_mask.sum(axis=-3) + 1e-8)

        else:
            # (batch_size, n, n, coefficients)
            combinations_sum = combinations.mean(axis=-3)

        combinations_norm = (combinations_sum ** 2).sum(-1)# .pow(0.5)

        h_combinations = self.post_norm_mlp(combinations_norm)
        # h_combinations = self.norm(h_combinations)
        return h_combinations, combinations

    def aggregate(self, h_e_mtx, mask=None):
        # h_e_mtx = self.mask_self(h_e_mtx)
        if mask is not None:
            h_e_mtx = h_e_mtx * jnp.expand_dims(mask, -1)
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

    def euclidean_attention(self, x_minus_xt_norm, mask=None):
        # (batch_size, n, n, 1)
        _x_minus_xt_norm = x_minus_xt_norm + 1e5 * jnp.expand_dims(jnp.eye(
            x_minus_xt_norm.shape[-2],
            x_minus_xt_norm.shape[-2],
        ), -1)

        if mask is not None:
            _x_minus_xt_norm = _x_minus_xt_norm + 1e5 * (1- jnp.expand_dims(mask, -1))

        att = jax.nn.softmax(
            -_x_minus_xt_norm * jnp.exp(self.log_gamma),
            axis=-2,
        )
        return att

    def semantic_attention(self, h_e_mtx, mask=None):
        # (batch_size, n, n, n_heads)
        att = self.semantic_attention_mlp(h_e_mtx)

        # (batch_size, n, n, n_heads)
        # att = att.view(*att.shape[:-1], self.n_heads)
        att = att - 1e5 * jnp.expand_dims(jnp.eye(
            att.shape[-2],
            att.shape[-2],
        ), -1)

        if mask is not None:
            att = att - 1e5 * (1 - jnp.expand_dims(mask, -1))

        att = jax.nn.softmax(att, axis=-2)
        return att

    def combined_attention(self, x_minus_xt_norm, h_e_mtx, mask=None):
        semantic_attention = self.semantic_attention(h_e_mtx, mask=mask)
        if self.cutoff is not None:
            euclidean_attention = self.cutoff(x_minus_xt_norm)
        else:
            euclidean_attention = 1.0

        combined_attention = euclidean_attention * semantic_attention
        if mask is not None:
            combined_attention = combined_attention - 1e5 * (1 - jnp.expand_dims(mask, -1))
        # combined_attention = jax.nn.softmax(combined_attention, axis=-2)
        combined_attention = combined_attention / combined_attention.sum(axis=-2, keepdims=True)
        
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
            he=None,
        ):

        x_minus_xt = get_x_minus_xt(x)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt)
        h_cat_ht = get_h_cat_ht(h)

        if he is not None:
            h_cat_ht = jnp.concatenate([h_cat_ht, he], -1)

        h_e_mtx = self.edge_model(h_cat_ht, x_minus_xt_norm)
        euclidean_attention, semantic_attention, combined_attention = self.combined_attention(x_minus_xt_norm, h_e_mtx, mask=mask)
        h_e_att = jnp.expand_dims(h_e_mtx, -1) * jnp.expand_dims(combined_attention, -2)
        h_e_att = jnp.reshape(h_e_att, h_e_att.shape[:-2] + (-1, ))
        h_combinations, delta_v = self.spatial_attention(h_e_att, x_minus_xt, x_minus_xt_norm, mask=mask)

        if not self.use_spatial_attention:
            h_combinations = jnp.zeros_like(h_combinations)
            delta_v = jnp.zeros_like(delta_v)

        # h_e_mtx = (h_e_mtx.unsqueeze(-1) * combined_attention.unsqueeze(-2)).flatten(-2, -1)
        h_e = self.aggregate(h_e_att, mask=mask)
        h = self.node_model(h, h_e, h_combinations)

        if self.update:
            if mask is not None:
                delta_v = self.v_mixing(delta_v.swapaxes(-1, -2)).swapaxes(-1, -2).sum(axis=(-2, -3))
                delta_v = delta_v / (mask.sum(-1, keepdims=True) + 1e-10)
            else:
                delta_v = self.v_mixing(delta_v.swapaxes(-1, -2)).swapaxes(-1, -2).mean(axis=(-2, -3))


            if v is not None:
                v = self.velocity_model(v, h)
            else:
                v = jnp.zeros_like(x)

            v = delta_v + v
            x = x + v


        return h, x, v

class SparseSAKELayer(SAKELayer):
    def spatial_attention(self, h_e_mtx, x_minus_xt, x_minus_xt_norm, idxs, num_segments=None):
        # (batch_size, n, n, n_coefficients)
        # coefficients = self.coefficients_mlp(h_e_mtx)# .unsqueeze(-1)
        coefficients = self.x_mixing(h_e_mtx)

        # (batch_size, n, n, 3)
        # x_minus_xt = x_minus_xt * euclidean_attention.mean(dim=-1, keepdim=True) / (x_minus_xt_norm + 1e-5)
        x_minus_xt = x_minus_xt / (x_minus_xt_norm + 1e-5) # ** 2

        # (batch_size, n, n, coefficients, 3)
        combinations = jnp.expand_dims(x_minus_xt, -2) * jnp.expand_dims(coefficients, -1)

        # dense shape: (batch_size, n, n, coefficients, 3)
        # dense: combinations_sum = combinations.mean(axis=-3)
        # sparse shape: (n_idxs, coefficients, 3)
        combinations_sum = jax.ops.segment_sum(combinations, idxs[..., 0], num_segments=num_segments)\
            /(jax.ops.segment_sum(jnp.ones_like(combinations), idxs[..., 0], num_segments=num_segments) + 1)
        
        # print("sparse", combinations_sum)

        combinations_norm = (combinations_sum ** 2).sum(-1)# .pow(0.5)

        h_combinations = self.post_norm_mlp(combinations_norm)
        return h_combinations, combinations

    def aggregate(self, h_e_mtx, idxs, num_segments):
        # dense: h_e = h_e_mtx.sum(axis=-2)
        h_e = jax.ops.segment_sum(h_e_mtx, idxs[..., 0], num_segments=num_segments)
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

    def semantic_attention(self, h_e_mtx, idxs, num_segments):
        # (batch_size, n, n, n_heads)
        att = self.semantic_attention_mlp(h_e_mtx)

        # (batch_size, n, n, n_heads)
        # att = att.view(*att.shape[:-1], self.n_heads)
        # att = att - 1e5 * jnp.expand_dims(jnp.eye(
        #     att.shape[-2],
        #     att.shape[-2],
        # ), -1)


        att = segment_softmax(att, idxs[..., 0], num_segments=num_segments)
        return att

    def combined_attention(self, x_minus_xt_norm, h_e_mtx, idxs, num_segments):
        semantic_attention = self.semantic_attention(h_e_mtx, idxs=idxs, num_segments=num_segments)
        if self.cutoff is not None:
            euclidean_attention = self.cutoff(x_minus_xt_norm)
        else:
            euclidean_attention = 1.0

        combined_attention = euclidean_attention * semantic_attention

        # combined_attention = combined_attention / combined_attention.sum(axis=-2, keepdims=True)
        combined_attention = combined_attention / jax.ops.segment_sum(combined_attention, idxs[..., 0], num_segments=num_segments)[idxs[..., 0]]
        
        return euclidean_attention, semantic_attention, combined_attention

    def velocity_model(self, v, h):
        v = self.velocity_mlp(h) * v
        return v

    def __call__(
            self,
            h,
            x,
            v=None,
            he=None,
            idxs=None,
        ):

        x_minus_xt = get_x_minus_xt_sparse(x, idxs)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt)

        h_cat_ht = get_h_cat_ht_sparse(h, idxs=idxs)

        if he is not None:
            h_cat_ht = jnp.concatenate([h_cat_ht, he], -1)

        h_e_mtx = self.edge_model(h_cat_ht, x_minus_xt_norm)
        euclidean_attention, semantic_attention, combined_attention = self.combined_attention(
            x_minus_xt_norm, h_e_mtx, idxs=idxs, num_segments=h.shape[-2],
        )
        h_e_att = jnp.expand_dims(h_e_mtx, -1) * jnp.expand_dims(combined_attention, -2)

        h_e_att = jnp.reshape(h_e_att, h_e_att.shape[:-2] + (-1, ))
        h_combinations, delta_v = self.spatial_attention(
            h_e_att, x_minus_xt, x_minus_xt_norm, idxs=idxs, num_segments=h.shape[-2],
        )

        if not self.use_spatial_attention:
            h_combinations = jnp.zeros_like(h_combinations)
            delta_v = jnp.zeros_like(delta_v)

        # h_e_mtx = (h_e_mtx.unsqueeze(-1) * combined_attention.unsqueeze(-2)).flatten(-2, -1)
        h_e = self.aggregate(h_e_att, idxs=idxs, num_segments=h.shape[-2])

        h = self.node_model(h, h_e, h_combinations)

        if self.update:
            delta_v = self.v_mixing(delta_v.swapaxes(-1, -2)).swapaxes(-1, -2).mean(-2)
            delta_v = jax.ops.segment_sum(delta_v, idxs[..., 0], num_segments=h.shape[-2])\
            /(jax.ops.segment_sum(jnp.ones_like(delta_v), idxs[..., 0], num_segments=h.shape[-2]) + 1)


            if v is not None:
                v = self.velocity_model(v, h)
            else:
                v = jnp.zeros_like(x)

            v = delta_v + v
            x = x + v

        return h, x, v

class EquivariantGraphConvolutionalLayer(nn.Module):
    out_features : int
    hidden_features : int
    activation : Callable = jax.nn.silu
    update : bool = False
    sigmoid : bool = False

    def setup(self):
        self.node_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.out_features),
                self.activation,
            ]
        )

        self.scaling_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(1, use_bias=False),
            ],
        )

        self.shifting_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(1, use_bias=False),
            ],
        )

        if self.sigmoid:
            self.edge_model = nn.Sequential(
                [
                    nn.Dense(1, use_bias=False),
                    jax.nn.sigmoid,
                ],
            )

    def aggregate(self, h_e_mtx, mask=None):
        # h_e_mtx = self.mask_self(h_e_mtx)
        if mask is not None:
            h_e_mtx = h_e_mtx * jnp.expand_dims(mask, -1)
        if self.sigmoid:
            h_e_weights = self.edge_model(h_e_mtx)
            h_e_mtx = h_e_weights * h_e_mtx
        h_e = h_e_mtx.sum(axis=-2)
        return h_e

    def node_model(self, h, h_e):
        out = jnp.concatenate([
                h,
                h_e,
            ],
            axis=-1)
        out = self.node_mlp(out)
        out = h + out
        return out

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
        h_e_mtx = jnp.concatenate([h_cat_ht, x_minus_xt_norm], axis=-1)
        h_e = self.aggregate(h_e_mtx, mask=mask)
        shift = self.shifting_mlp(h_e_mtx).sum(-2)
        scale = self.scaling_mlp(h)

        if self.update:
            v = v * scale + shift
            x = x + v
        h = self.node_model(h, h_e)
        return h, x, v


class EquivariantGraphConvolutionalLayerWithSmearing(nn.Module):
    out_features : int
    hidden_features : int
    activation : Callable = jax.nn.silu
    update : bool = False
    sigmoid: bool = True

    def setup(self):
        self.edge_model = ContinuousFilterConvolutionWithConcatenation(self.hidden_features)

        self.node_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(self.out_features),
                self.activation,
            ]
        )

        self.scaling_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(1, use_bias=False),
            ],
        )

        self.shifting_mlp = nn.Sequential(
            [
                nn.Dense(self.hidden_features),
                self.activation,
                nn.Dense(1, use_bias=False),
            ],
        )

        if self.sigmoid:
            self.edge_att = nn.Sequential(
                [
                    nn.Dense(1, use_bias=False),
                    jax.nn.sigmoid,
                ],
            )



    def aggregate(self, h_e_mtx, mask=None):
        # h_e_mtx = self.mask_self(h_e_mtx)
        if mask is not None:
            h_e_mtx = h_e_mtx * jnp.expand_dims(mask, -1)
        if self.sigmoid:
            h_e_weights = self.edge_att(h_e_mtx)
            h_e_mtx = h_e_weights * h_e_mtx
        h_e = h_e_mtx.sum(axis=-2)
        return h_e

    def node_model(self, h, h_e):
        out = jnp.concatenate([
                h,
                h_e,
            ],
            axis=-1)
        out = self.node_mlp(out)
        out = h + out
        return out

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
        h_e = self.aggregate(h_e_mtx, mask=mask)
        shift = self.shifting_mlp(h_e_mtx).sum(-2)
        scale = self.scaling_mlp(h)

        if self.update:
            v = v * scale + shift
            x = x + v
        h = self.node_model(h, h_e)
        return h, x, v
