import pytest

# def test_import_and_init():
#     import jax
#     import jax.numpy as jnp
#     from sake.models import RigidDocking
#     model = RigidDocking(8, 8)
#     x0 = jax.random.normal(jax.random.PRNGKey(1984), shape=(4, 3))
#     h0 = jax.random.normal(jax.random.PRNGKey(1984), shape=(4, 6))
#     x1 = jax.random.normal(jax.random.PRNGKey(1989), shape=(4, 3))
#     h1 = jax.random.normal(jax.random.PRNGKey(1989), shape=(4, 6))
#     params = model.init(jax.random.PRNGKey(2666), h0, x0, h1, x1)
#
# def test_change_of_basis():
#     import jax
#     import jax.numpy as jnp
#     from sake.models import RigidDocking
#     model = RigidDocking(8, 8)
#     get_change_of_basis_matrix = model.change_of_coordinates_matrix
#     x_axis = jax.random.normal(jax.random.PRNGKey(1984), shape=(2, 3))
#     y_axis = jax.random.normal(jax.random.PRNGKey(1989), shape=(2, 3))
#
#     # x_axis = jnp.array([1.0, 0.0, 0.0])
#     # y_axis = jnp.array([0.0, 1.0, 0.0])
#     # x_axis = jnp.stack([x_axis, x_axis], 0)
#     # y_axis = jnp.stack([y_axis, y_axis], 0)
#
#     x_axis = x_axis / jnp.linalg.norm(x_axis, axis=-1, keepdims=True)
#     y_axis = y_axis / jnp.linalg.norm(y_axis, axis=-1, keepdims=True)
#     p = get_change_of_basis_matrix(x_axis, y_axis)
#
#     _x = jax.vmap(jnp.matmul)(x_axis, p)
#     _y = jax.vmap(jnp.matmul)(y_axis, p)
#
#     assert jnp.allclose(_x, jnp.array([1.0, 0.0, 0.0]), atol=1e-3)
#     assert jnp.allclose(_y, jnp.array([0.0, 1.0, 0.0]), atol=1e-3)

def test_commute():
    import jax
    import jax.numpy as jnp
    from sake.models import RigidDocking
    model = RigidDocking(8, 8)
    x0 = jax.random.normal(jax.random.PRNGKey(1984), shape=(4, 3))
    h0 = jax.random.normal(jax.random.PRNGKey(1984), shape=(4, 6))
    x1 = jax.random.normal(jax.random.PRNGKey(1989), shape=(4, 3))
    h1 = jax.random.normal(jax.random.PRNGKey(1989), shape=(4, 6))
    params = model.init(jax.random.PRNGKey(2666), h0, x0, h1, x1)

    _h0, _x0, _h1, _x1 = model.apply(params, h0, x0, h1, x1, method=model.match)
    __h1, __x1, __h0, __x0 = model.apply(params, h1, x1, h0, x0, method=model.match)

    assert jnp.allclose(_x0, __x0)
