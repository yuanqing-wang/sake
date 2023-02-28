# def test_sparse_sake_layer():
#     import jax
#     import jax.numpy as jnp
#     import sake
#     model = sake.layers.SparseSAKELayer(16, 16)
#     x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
#     h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
#     idxs = jnp.zeros((5, 2), jnp.int32)
#     init_params = model.init(jax.random.PRNGKey(2046), h, x, idxs=idxs)
#     h, x, v = model.apply(init_params, h, x, idxs=idxs)
#     assert h.shape == (5, 16)
#     assert x.shape == (5, 3)
#     assert v.shape == (5, 3)


def test_get_x_minus_xt_consistent():
    import sake
    import jax
    import jax.numpy as jnp
    from sake.functional import (
        get_x_minus_xt, get_x_minus_xt_norm, get_h_cat_ht,
        get_x_minus_xt_sparse, get_h_cat_ht_sparse,
    )

    def allclose_dense_sparse(dense, sparse, idxs):
        return jnp.allclose(
            dense[idxs[:, 0], idxs[:, 1]], sparse,
        )
    
    from flax import linen as nn
    class Accessible(nn.Module):
        base: nn.Module

        def edge_model(self, *args, **kwargs):
            return self.base.edge_model(*args, **kwargs)
        
        def x_mixing(self, *args, **kwargs):
            return self.base.x_mixing(*args, **kwargs)

    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    idxs = jnp.stack(
        jnp.where(
            jnp.eye(x.shape[0]) < 1
        ),
        axis=-1,
    )

    model_dense = sake.layers.DenseSAKELayer(16, 16, n_heads=1)
    model_sparse = sake.layers.SparseSAKELayer(16, 16, n_heads=1)
    
    accesible_dense = Accessible(model_dense)
    accesible_sparse = Accessible(model_sparse)

    params = model_dense.init(jax.random.PRNGKey(2046), h, x)

    x_minus_xt_dense = get_x_minus_xt(x)
    x_minus_xt_sparse = get_x_minus_xt_sparse(x, idxs)
    assert allclose_dense_sparse(
        x_minus_xt_dense, x_minus_xt_sparse, idxs
    )

    x_minus_xt_norm_dense = get_x_minus_xt_norm(x_minus_xt_dense)
    x_minus_xt_norm_sparse = get_x_minus_xt_norm(x_minus_xt_sparse)
    assert allclose_dense_sparse(
        x_minus_xt_norm_dense, x_minus_xt_norm_sparse, idxs,
    )

    h_cat_ht_dense = get_h_cat_ht(h)
    h_cat_ht_sparse = get_h_cat_ht_sparse(h, idxs=idxs)
    assert allclose_dense_sparse(
        h_cat_ht_dense, h_cat_ht_sparse, idxs,
    )

    h_e_mtx_dense = accesible_dense.apply(
        {"params": {"base": params["params"]}}, h_cat_ht_dense, x_minus_xt_norm_dense, method=accesible_dense.edge_model,
    )
    h_e_mtx_sparse = accesible_sparse.apply(
        {"params": {"base": params["params"]}}, h_cat_ht_sparse, x_minus_xt_norm_sparse, method=accesible_dense.edge_model,
    )
    assert allclose_dense_sparse(
        h_e_mtx_dense, h_e_mtx_sparse, idxs,
    )

    semantic_attention_dense = model_dense.apply(
        params, h_e_mtx_dense, method=model_dense.semantic_attention,
    )

    semantic_attention_sparse = model_sparse.apply(
        params, h_e_mtx_sparse, idxs=idxs, num_segments=5, method=model_sparse.semantic_attention,
    )

    assert allclose_dense_sparse(
        semantic_attention_dense, semantic_attention_sparse, idxs
    )


    _, __, combined_attention_dense = model_dense.apply(
        params, x_minus_xt_norm_dense, h_e_mtx_dense, method=model_dense.combined_attention,
    )

    _, __, combined_attention_sparse = model_sparse.apply(
        params, x_minus_xt_norm_sparse, h_e_mtx_sparse, idxs=idxs, num_segments=5, method=model_sparse.combined_attention,
    )

    assert allclose_dense_sparse(
        combined_attention_dense, combined_attention_sparse, idxs
    )

    h_e_att_dense = jnp.expand_dims(h_e_mtx_dense, -1) * jnp.expand_dims(combined_attention_dense, -2)
    h_e_att_sparse = jnp.expand_dims(h_e_mtx_sparse, -1) * jnp.expand_dims(combined_attention_sparse, -2)
    h_e_att_dense = jnp.reshape(h_e_att_dense, h_e_att_dense.shape[:-2] + (-1, ))
    h_e_att_sparse = jnp.reshape(h_e_att_sparse, h_e_att_sparse.shape[:-2] + (-1, ))
    assert allclose_dense_sparse(
        h_e_att_dense, h_e_att_sparse, idxs
    )

    h_combinations_dense, delta_v_dense = model_dense.apply(
        params, h_e_att_dense, x_minus_xt_dense, x_minus_xt_norm_dense, method=model_dense.spatial_attention,
    )
    h_combinations_sparse, delta_v_sparse = model_sparse.apply(
        params, h_e_att_sparse, x_minus_xt_sparse, x_minus_xt_norm_sparse, idxs=idxs, num_segments=5, method=model_sparse.spatial_attention,
    )
    assert jnp.allclose(h_combinations_dense, h_combinations_sparse)
    assert allclose_dense_sparse(delta_v_dense, delta_v_sparse, idxs)

    h_e_dense = model_dense.apply(
        params, h_e_att_dense, method=model_dense.aggregate
    )
    h_e_sparse = model_sparse.apply(
        params, h_e_att_sparse, idxs=idxs, num_segments=5, method=model_sparse.aggregate,
    )
    assert jnp.allclose(h_e_dense, h_e_sparse)

    h_dense = model_dense.apply(
        params, h, h_e_dense, h_combinations_dense, method=model_dense.node_model
    )
    h_sparse = model_sparse.apply(
        params, h, h_e_sparse, h_combinations_sparse, method=model_sparse.node_model
    )
    assert jnp.allclose(h_dense, h_sparse)

def test_dense_sparse_consistent():
    import jax
    import jax.numpy as jnp
    import sake
    model_dense = sake.layers.DenseSAKELayer(16, 16, cutoff=sake.utils.cosine_cutoff)
    model = sake.layers.SparseSAKELayer(16, 16, cutoff=sake.utils.cosine_cutoff)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    init_params = model_dense.init(jax.random.PRNGKey(2046), h, x)
    idxs = jnp.stack(
        jnp.where(
            jnp.eye(x.shape[0]) < 1
        ),
        axis=1,
    )

    h_dense, x_dense, v_dense = model_dense.apply(init_params, h, x)
    h, x, v = model.apply(init_params, h, x, idxs=idxs)

    assert h.shape == h_dense.shape == (5, 16)
    assert x.shape == x_dense.shape == (5, 3)
    assert v.shape == v_dense.shape == (5, 3)

    assert jnp.allclose(h, h_dense)


