import pytest

def test_odeflow():
    import jax
    import jax.numpy as jnp
    import sake
    from functools import partial
    key = jax.random.PRNGKey(2666)
    model = sake.models.DenseSAKEModel(hidden_features=4, depth=2, out_features=1)
    x = jax.random.normal(key=key, shape=(2, 5, 3))
    t = jax.random.normal(key=key, shape=(2, 5, 1))
    params = model.init(key, t, x)
    dynamics = partial(sake.flows.ODEFlow.dynamics, model, params)
    assert dynamics(x, t).shape == (2, 5, 3)

    integrate = partial(sake.flows.ODEFlow.integrate, dynamics)
    assert integrate(x).shape == (2, 5, 3)

    jacobian = partial(sake.flows.ODEFlow.jacobian, integrate)
    assert jacobian(x).shape == (2, 5, 3, 5, 3)

    trace = partial(sake.flows.ODEFlow.trace, jacobian)
    assert trace(x).shape == (2,)

    call = partial(sake.flows.ODEFlow(), model, params)
    res, trace = call(x)
    assert res.shape == (2, 5, 3)
    assert trace.shape == (2, )



if __name__ == "__main__":
    test_odeflow()
