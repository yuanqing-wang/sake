import pytest
import jax
import jax.numpy as jnp
import numpy as onp

@pytest.fixture
def _equivariance_test_utils():
    h0 = jax.random.normal(
        key=jax.random.PRNGKey(0),
        shape=(5, 7),
    )

    x0 = jax.random.normal(
        key=jax.random.PRNGKey(1),
        shape=(5, 3),
    )

    x_translation = jax.random.normal(
        key=jax.random.PRNGKey(2),
        shape=(1, 3),
    )

    v0 = jax.random.normal(
        key=jax.random.PRNGKey(3),
        shape=(5, 3),
    )

    translation = lambda x: x + x_translation

    import math
    alpha = onp.random.uniform(-math.pi, math.pi)
    beta = onp.random.uniform(-math.pi, math.pi)
    gamma = onp.random.uniform(-math.pi, math.pi)

    rz = jnp.array(
        [
            [math.cos(alpha), -math.sin(alpha), 0],
            [math.sin(alpha),  math.cos(alpha), 0],
            [0,                0,               1],
        ]
    )

    ry = jnp.array(
        [
            [math.cos(beta),   0,               math.sin(beta)],
            [0,                1,               0],
            [-math.sin(beta),  0,               math.cos(beta)],
        ]
    )

    rx = jnp.array(
        [
            [1,                0,               0],
            [0,                math.cos(gamma), -math.sin(gamma)],
            [0,                math.sin(gamma), math.cos(gamma)],
        ]
    )

    rotation = lambda x: x @ rz @ ry @ rx

    alpha = onp.random.uniform(-math.pi, math.pi)
    beta = onp.random.uniform(-math.pi, math.pi)
    gamma = onp.random.uniform(-math.pi, math.pi)
    v = jnp.array([[alpha, beta, gamma]])
    v /= (v ** 2).sum() ** 0.5

    p = jnp.eye(3) - 2 * v.T @ v

    reflection = lambda x: x @ p

    return h0, x0, v0, translation, rotation, reflection
