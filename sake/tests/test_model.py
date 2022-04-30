import pytest

def test_dense_sake_layer():
    import jax
    import jax.numpy as jnp
    import sake
    model = sake.models.DenseSAKEModel(16, 16)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    init_params = model.init(jax.random.PRNGKey(2046), h, x)
    h, x, v = model.apply(init_params, h, x)
    assert h.shape == (5, 16)
    assert x.shape == (5, 3)
    assert v.shape == (5, 3)
