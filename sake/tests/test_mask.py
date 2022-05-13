import pytest\

# def test_distance():
#     import jax
#     import jax.numpy as jnp
#     import sake
#     x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
#     x_mask = jnp.concatenate([x, jnp.ones((1, 3))], axis=0)
#     mask = jnp.concatenate([jnp.ones(5), jnp.zeros(1)], axis=0)
#     mask = jnp.expand_dims(mask, 0) * jnp.expand_dims(mask, 1)
#     x_minus_xt_norm_mask = sake.functional.get_x_minus_xt_norm(
#         sake.functional.get_x_minus_xt(
#             x_mask
#         )
#     )[:5, :5, :]
#
#     x_minus_xt_norm = sake.functional.get_x_minus_xt_norm(
#         sake.functional.get_x_minus_xt(
#             x
#         )
#     )
#
#     assert jnp.allclose(
#         x_minus_xt_norm_mask,
#         x_minus_xt_norm,
#     )
#
# def test_concat():
#     import jax
#     import jax.numpy as jnp
#     import sake
#     x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
#     x_mask = jnp.concatenate([x, jnp.ones((1, 3))], axis=0)
#     mask = jnp.concatenate([jnp.ones(5), jnp.zeros(1)], axis=0)
#     mask = jnp.expand_dims(mask, 0) * jnp.expand_dims(mask, 1)
#     h_cat_ht_mask = sake.functional.get_h_cat_ht(x_mask)[:5, :5, :]
#     h_cat_ht = sake.functional.get_h_cat_ht(x)
#
#     assert jnp.allclose(
#         h_cat_ht_mask,
#         h_cat_ht,
#     )

def test_euclidean_attention():
    import jax
    import jax.numpy as jnp
    import sake
    model = sake.layers.DenseSAKELayer(16, 16)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    init_params = model.init(jax.random.PRNGKey(2046), h, x)
    x_minus_xt_norm = sake.functional.get_x_minus_xt_norm(
        sake.functional.get_x_minus_xt(x)
    )
    att = model.apply(init_params, x_minus_xt_norm, method=model.euclidean_attention)

    x_mask = jnp.concatenate([x, jnp.ones((1, 3))], axis=0)
    h_mask = jnp.concatenate([h, jnp.ones((1, 16))], axis=0)
    mask = jnp.concatenate([jnp.ones(5), jnp.zeros(1)], axis=0)
    mask = jnp.expand_dims(mask, 0) * jnp.expand_dims(mask, 1)
    x_minus_xt_norm_mask = sake.functional.get_x_minus_xt_norm(
        sake.functional.get_x_minus_xt(x_mask)
    )
    att_mask = model.apply(init_params, x_minus_xt_norm_mask, mask=mask, method=model.euclidean_attention)[:5, :5, :]

    assert jnp.allclose(
        att,
        att_mask,
    )

def test_semantic_attention():
    import jax
    import jax.numpy as jnp
    import sake
    model = sake.layers.DenseSAKELayer(16, 16)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    init_params = model.init(jax.random.PRNGKey(2046), h, x)
    h_cat_ht = sake.functional.get_h_cat_ht(h)[:, :, :16]
    att = model.apply(init_params, h_cat_ht, method=model.semantic_attention)

    x_mask = jnp.concatenate([x, jnp.ones((1, 3))], axis=0)
    h_mask = jnp.concatenate([h, jnp.ones((1, 16))], axis=0)
    mask = jnp.concatenate([jnp.ones(5), jnp.zeros(1)], axis=0)
    mask = jnp.expand_dims(mask, 0) * jnp.expand_dims(mask, 1)
    h_cat_ht_mask = sake.functional.get_h_cat_ht(h_mask)[:, :, :16]
    att_mask = model.apply(init_params, h_cat_ht_mask, mask=mask, method=model.semantic_attention)[:5, :5, :]

    assert jnp.allclose(
        att,
        att_mask,
    )

def test_combined_attention():
    import jax
    import jax.numpy as jnp
    import sake
    model = sake.layers.DenseSAKELayer(16, 16)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    init_params = model.init(jax.random.PRNGKey(2046), h, x)
    h_cat_ht = sake.functional.get_h_cat_ht(h)[:, :, :16]
    x_minus_xt_norm = sake.functional.get_x_minus_xt_norm(
        sake.functional.get_x_minus_xt(x)
    )
    euclidean_attention, semantic_attention, combined_attention =\
        model.apply(init_params, x_minus_xt_norm, h_cat_ht, method=model.combined_attention)

    x_mask = jnp.concatenate([x, jnp.ones((1, 3))], axis=0)
    h_mask = jnp.concatenate([h, jnp.ones((1, 16))], axis=0)
    mask = jnp.concatenate([jnp.ones(5), jnp.zeros(1)], axis=0)
    mask = jnp.expand_dims(mask, 0) * jnp.expand_dims(mask, 1)
    h_cat_ht_mask = sake.functional.get_h_cat_ht(h_mask)[:, :, :16]
    x_minus_xt_norm_mask = sake.functional.get_x_minus_xt_norm(
        sake.functional.get_x_minus_xt(x_mask)
    )
    euclidean_attention_mask, semantic_attention_mask, combined_attention_mask =\
        model.apply(init_params, x_minus_xt_norm_mask, h_cat_ht_mask, mask=mask, method=model.combined_attention)

    euclidean_attention_mask = euclidean_attention_mask[:5, :5, :]
    semantic_attention_mask = semantic_attention_mask[:5, :5, :]
    combined_attention_mask = combined_attention_mask[:5, :5, :]

    assert jnp.allclose(
        euclidean_attention,
        euclidean_attention_mask,
    )

    assert jnp.allclose(
        semantic_attention,
        semantic_attention_mask,
    )

    assert jnp.allclose(
        combined_attention,
        combined_attention_mask,
    )


def test_spatial_attention():
    import jax
    import jax.numpy as jnp
    import sake
    model = sake.layers.DenseSAKELayer(16, 16)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    init_params = model.init(jax.random.PRNGKey(2046), h, x)
    h_cat_ht = sake.functional.get_h_cat_ht(h)[:, :, :16]
    x_minus_xt = sake.functional.get_x_minus_xt(x)
    x_minus_xt_norm = sake.functional.get_x_minus_xt_norm(x_minus_xt)
    euclidean_attention, semantic_attention, combined_attention =\
        model.apply(init_params, x_minus_xt_norm, h_cat_ht, method=model.combined_attention)
    h_e_att = jnp.expand_dims(h_cat_ht, -1) * jnp.expand_dims(combined_attention, -2)
    h_e_att = jnp.reshape(h_e_att, h_e_att.shape[:-2] + (-1, ))[:, :, :64]
    h_combinations, combinations = model.apply(init_params, h_e_att, x_minus_xt, x_minus_xt_norm, method=model.spatial_attention)

    x_mask = jnp.concatenate([x, jnp.ones((1, 3))], axis=0)
    h_mask = jnp.concatenate([h, jnp.ones((1, 16))], axis=0)
    mask = jnp.concatenate([jnp.ones(5), jnp.zeros(1)], axis=0)
    mask = jnp.expand_dims(mask, 0) * jnp.expand_dims(mask, 1)
    h_cat_ht_mask = sake.functional.get_h_cat_ht(h_mask)[:, :, :16]
    x_minus_xt_mask = sake.functional.get_x_minus_xt(x_mask)
    x_minus_xt_norm_mask = sake.functional.get_x_minus_xt_norm(x_minus_xt_mask)
    euclidean_attention_mask, semantic_attention_mask, combined_attention_mask =\
        model.apply(init_params, x_minus_xt_norm_mask, h_cat_ht_mask, mask=mask, method=model.combined_attention)
    h_e_att_mask = jnp.expand_dims(h_cat_ht_mask, -1) * jnp.expand_dims(combined_attention_mask, -2)
    h_e_att_mask = jnp.reshape(h_e_att_mask, h_e_att_mask.shape[:-2] + (-1, ))[:, :, :64]
    h_combinations_mask, combinations_mask  = model.apply(init_params, h_e_att_mask, x_minus_xt_mask, x_minus_xt_norm_mask, mask=mask, method=model.spatial_attention)

    assert jnp.allclose(
        combinations,
        combinations_mask[:5, :5, :],
    )

    assert jnp.allclose(
        h_combinations,
        h_combinations_mask[:5],
    )

def test_aggregate():
    import jax
    import jax.numpy as jnp
    import sake
    model = sake.layers.DenseSAKELayer(16, 16)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    init_params = model.init(jax.random.PRNGKey(2046), h, x)
    h_cat_ht = sake.functional.get_h_cat_ht(h)
    h_e = model.apply(init_params, h_cat_ht, method=model.aggregate)

    x_mask = jnp.concatenate([x, jnp.ones((1, 3))], axis=0)
    h_mask = jnp.concatenate([h, jnp.ones((1, 16))], axis=0)
    mask = jnp.concatenate([jnp.ones(5), jnp.zeros(1)], axis=0)
    mask = jnp.expand_dims(mask, 0) * jnp.expand_dims(mask, 1)
    h_cat_ht_mask = sake.functional.get_h_cat_ht(h_mask)
    h_e_mask = model.apply(init_params, h_cat_ht_mask, mask=mask, method=model.aggregate)[:5]
    assert jnp.allclose(
        h_e,
        h_e_mask,
    )

def test_dense_sake_layer():
    import jax
    import jax.numpy as jnp
    import sake
    model = sake.layers.DenseSAKELayer(16, 16)
    x0 = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h0 = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    init_params = model.init(jax.random.PRNGKey(2046), h0, x0)
    h, x, v = model.apply(init_params, h0, x0)

    x_mask = jnp.concatenate([x0, jnp.ones((1, 3))], axis=0)
    h_mask = jnp.concatenate([h0, jnp.ones((1, 16))], axis=0)
    mask = jnp.concatenate([jnp.ones(5), jnp.zeros(1)], axis=0)
    mask = jnp.expand_dims(mask, 0) * jnp.expand_dims(mask, 1)
    h_mask, x_mask, v_mask = model.apply(init_params, h_mask, x_mask, mask=mask)

    assert jnp.allclose(h, h_mask[:-1])
    assert jnp.allclose(x, x_mask[:-1])
    assert jnp.allclose(v, v_mask[:-1])

def test_dense_sake_model():
    import jax
    import jax.numpy as jnp
    import sake
    model = sake.models.DenseSAKEModel(16, 16)
    x0 = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h0 = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    init_params = model.init(jax.random.PRNGKey(2046), h0, x0)
    h, x, v = model.apply(init_params, h0, x0)

    x_mask = jnp.concatenate([x0, jnp.ones((1, 3))], axis=0)
    h_mask = jnp.concatenate([h0, jnp.ones((1, 16))], axis=0)
    mask = jnp.concatenate([jnp.ones(5), jnp.zeros(1)], axis=0)
    mask = jnp.expand_dims(mask, 0) * jnp.expand_dims(mask, 1)
    h_mask, x_mask, v_mask = model.apply(init_params, h_mask, x_mask, mask=mask)

    assert jnp.allclose(h, h_mask[:-1])
    assert jnp.allclose(x, x_mask[:-1])
    assert jnp.allclose(v, v_mask[:-1])
