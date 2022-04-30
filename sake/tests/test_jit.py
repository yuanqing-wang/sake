import pytest
def test_dense_sake_layer():
    import jax
    import jax.numpy as jnp
    import sake
    model = sake.layers.DenseSAKELayer(16, 16)
    x = jax.random.normal(key=jax.random.PRNGKey(2666), shape=(5, 3))
    h = jax.random.uniform(key=jax.random.PRNGKey(1984), shape=(5, 16))
    init_params = model.init(jax.random.PRNGKey(2046), h, x)

    @jax.jit
    def fn(x, h):
        h, x, v = model.apply(init_params, h, x)
        return h, x, v

    fn(x, h)
    
