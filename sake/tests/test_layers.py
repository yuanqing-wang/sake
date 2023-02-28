import pytest

# def test_exp_normal_smearing():
#     import jax
#     import jax.numpy as jnp
#     import sake
#     model = sake.utils.ExpNormalSmearing()
#     key = jax.random.PRNGKey(2666)
#     x = jax.random.normal(key=key, shape=(5, 5, 1))
#     init_params = model.init(key, x)
#     out = model.apply(init_params, x)
#     assert out.shape == x.shape[:-1] + (model.num_rbf,)

# def test_cfc_with_concatenation():
#     import jax
#     import jax.numpy as jnp
#     import sake
#     model = sake.layers.ContinuousFilterConvolutionWithConcatenation(16)
#     x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 5, 1))
#     h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 5, 4))
#     init_params = model.init(jax.random.PRNGKey(2046), h, x)
#     h = model.apply(init_params, h, x)
#     assert h.shape == (5, 5, 16)

# def test_dense_sake_layer():
#     import jax
#     import jax.numpy as jnp
#     import sake
#     model = sake.layers.DenseSAKELayer(16, 16)
#     x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
#     h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
#     init_params = model.init(jax.random.PRNGKey(2046), h, x)
#     h, x, v = model.apply(init_params, h, x)
#     assert h.shape == (5, 16)
#     assert x.shape == (5, 3)
#     assert v.shape == (5, 3)

def test_sparse_sake_layer():
    import jax
    import jax.numpy as jnp
    import sake
    model = sake.layers.SparseSAKELayer(16, 16)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    idxs = jnp.zeros((5, 2), jnp.int32)
    init_params = model.init(jax.random.PRNGKey(2046), h, x, idxs=idxs)
    h, x, v = model.apply(init_params, h, x, idxs=idxs)
    assert h.shape == (5, 16)
    assert x.shape == (5, 3)
    assert v.shape == (5, 3)

def test_dense_sparse_consistent():
    import jax
    import jax.numpy as jnp
    import sake
    model = sake.layers.SparseSAKELayer(16, 16)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    idxs = jnp.zeros((5, 2), jnp.int32)
    init_params = model.init(jax.random.PRNGKey(2046), h, x, idxs=idxs)
    h, x, v = model.apply(init_params, h, x, idxs=idxs)
    assert h.shape == (5, 16)
    assert x.shape == (5, 3)
    assert v.shape == (5, 3)
    