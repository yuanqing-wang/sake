import jax
import jax.numpy as jnp

EPSILON = 1e-5
INF = 1e5

def get_x_minus_xt(x):
    return jnp.expand_dims(x, -3) - jnp.expand_dims(x, -2)

def get_x_minus_xt_norm(
    x_minus_xt,
    epsilon: float=EPSILON,
):
    x_minus_xt_norm = (
        jax.nn.relu((x_minus_xt ** 2).sum(axis=-1, keepdims=True))
        + epsilon
    ) ** 0.5

    return x_minus_xt_norm

# def get_h_cat_ht(h):
#     n_nodes = h.shape[-2]
#     h_cat_ht = jnp.concatenate(
#         [
#             jnp.repeat(jnp.expand_dims(h, -3), n_nodes, -3),
#             jnp.repeat(jnp.expand_dims(h, -2), n_nodes, -2)
#         ],
#         axis=-1
#     )
#
#     return h_cat_ht

def get_h_cat_ht(h):
    n_nodes = h.shape[-2]
    h_shape = (*h.shape[:-2], n_nodes, n_nodes, h.shape[-1])
    h_cat_ht = jnp.concatenate(
        [
            jnp.broadcast_to(jnp.expand_dims(h, -3), h_shape),
            jnp.broadcast_to(jnp.expand_dims(h, -2), h_shape),
        ],
        axis=-1,
    )

    return h_cat_ht
