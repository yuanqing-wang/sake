import pytest

def test_layer_equivariance(_equivariance_test_utils):
    import jax
    import jax.numpy as jnp
    import sake
    h0, x0, v0, translation, rotation, reflection = _equivariance_test_utils
    model = sake.layers.DenseSAKELayer(7, 7)

    init_params = model.init(jax.random.PRNGKey(2666), h0, x0, v0)

    h_original, x_original, v_original = model.apply(init_params, h0, x0, v0)
    h_translation, x_translation, v_translation = model.apply(init_params, h0, translation(x0), v0)
    h_rotation, x_rotation, v_rotation = model.apply(init_params, h0, rotation(x0), rotation(v0))
    h_reflection, x_reflection, v_reflection = model.apply(init_params, h0, reflection(x0), reflection(v0))

    assert jnp.allclose(h_translation, h_original)
    assert jnp.allclose(h_rotation, h_original)
    assert jnp.allclose(h_reflection, h_original)

    assert jnp.allclose(x_translation, translation(x_original))
    assert jnp.allclose(x_rotation, rotation(x_original))
    assert jnp.allclose(x_reflection, reflection(x_original))

def test_model_equivariance(_equivariance_test_utils):
    import jax
    import jax.numpy as jnp
    import sake
    h0, x0, v0, translation, rotation, reflection = _equivariance_test_utils
    model = sake.models.DenseSAKEModel(7, 7)

    init_params = model.init(jax.random.PRNGKey(2666), h0, x0, v0)

    h_original, x_original, v_original = model.apply(init_params, h0, x0, v0)
    h_translation, x_translation, v_translation = model.apply(init_params, h0, translation(x0), v0)
    h_rotation, x_rotation, v_rotation = model.apply(init_params, h0, rotation(x0), rotation(v0))
    h_reflection, x_reflection, v_reflection = model.apply(init_params, h0, reflection(x0), reflection(v0))

    assert jnp.allclose(h_translation, h_original)
    assert jnp.allclose(h_rotation, h_original)
    assert jnp.allclose(h_reflection, h_original)

    assert jnp.allclose(x_translation, translation(x_original))
    assert jnp.allclose(x_rotation, rotation(x_original))
    assert jnp.allclose(x_reflection, reflection(x_original))
