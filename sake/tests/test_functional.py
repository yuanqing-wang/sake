import pytest

def test_x_minus_xt():
    import jax
    import jax.numpy as jnp
    import sake
    key = jax.random.PRNGKey(2666)
    x = jax.random.normal(key=key, shape=(5, 3))
    x_minus_xt = sake.functional.get_x_minus_xt(x)
    assert x_minus_xt.shape == (5, 5, 3)

def test_x_minus_xt_norm():
    import jax
    import jax.numpy as jnp
    import sake
    key = jax.random.PRNGKey(2666)
    x = jax.random.normal(key=key, shape=(5, 3))
    x_minus_xt = sake.functional.get_x_minus_xt(x)
    x_minus_xt_norm = sake.functional.get_x_minus_xt_norm(x_minus_xt)
    assert x_minus_xt_norm.shape == (5, 5, 1)

def test_h_cat_ht():
    import jax
    import jax.numpy as jnp
    import sake
    key = jax.random.PRNGKey(2666)
    h = jax.random.normal(key=key, shape=(5, 3))
    h_cat_ht = sake.functional.get_h_cat_ht(h)
    assert h_cat_ht.shape == (5, 5, 6)
