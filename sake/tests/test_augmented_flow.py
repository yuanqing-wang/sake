import pytest

def test_augmented_flow_layer():
    import jax
    import jax.numpy as jnp
    import sake
    from functools import partial
    model = sake.flows.AugmentedFlowLayer()
    h = jax.random.normal(key=jax.random.PRNGKey(0), shape=(5, 1))
    x = sake.flows.CenteredGaussian().sample(key=jax.random.PRNGKey(1), shape=(5, 3))
    v = sake.flows.CenteredGaussian().sample(key=jax.random.PRNGKey(2), shape=(5, 3))
    params = model.init(jax.random.PRNGKey(3), h, x, v)
    _x, _v, log_det = model.apply(params, h, x, v)

    assert jnp.allclose(_x.mean(-2), 0.0, atol=1e-5, rtol=10000)
    assert jnp.allclose(_v.mean(-2), 0.0, atol=1e-5, rtol=10000)

    def fn(z):
        x, v = jnp.split(z, 2, axis=-1)
        _x, _v, _ = model.apply(params, h, x, v)
        return jnp.concatenate([_x, _v], axis=-1)

    z = jnp.concatenate([x, v], axis=-1)
    jac = jax.jacrev(fn)(z).reshape([30, 30])
    _, log_det_auto = jnp.linalg.slogdet(jac)
    assert jnp.allclose(log_det_auto, log_det)
