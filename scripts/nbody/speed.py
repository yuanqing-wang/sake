import jax
import jax.numpy as jnp
import sake

def run():
    q_tr = jnp.load("charges_train_charged5_initvel1.npy")
    q_vl = jnp.load("charges_valid_charged5_initvel1.npy")
    q_te = jnp.load("charges_test_charged5_initvel1.npy")
    x_tr = jnp.load("loc_train_charged5_initvel1.npy")
    x_vl = jnp.load("loc_valid_charged5_initvel1.npy")
    x_te = jnp.load("loc_test_charged5_initvel1.npy")
    v_tr = jnp.load("vel_train_charged5_initvel1.npy")
    v_vl = jnp.load("vel_valid_charged5_initvel1.npy")
    v_te = jnp.load("vel_test_charged5_initvel1.npy")

    x = x_tr.reshape((30, 100, *x_tr.shape[1:])) 
    q = q_tr.reshape((30, 100, *q_tr.shape[1:]))
    v = v_tr.reshape((30, 100, *v_tr.shape[1:]))
    
    
    # x = x_tr[:100]
    # q = q_tr[:100]
    # v = v_tr[:100]

    q = jnp.expand_dims(q, -3)
    q = jnp.repeat(q, x.shape[-3], -3)

    x = jnp.swapaxes(x, -2, -1)
    v = jnp.swapaxes(v, -2, -1)
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    h = jnp.concatenate([q, v_norm], axis=-1)

    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=4,
        update=True,
    )

    params = model.init(jax.random.PRNGKey(2666), h[0], x[0], v[0])

    @jax.jit
    def forward(h, x, v):
        for idx in range(30):
            _h, _x, _v = h[idx], x[idx], v[idx]
            _h, _x, _v = model.apply(params, _h, _x, _v)
        return _x

    for _ in range(1):
        jax.block_until_ready(forward(h, x, v))

    import time
    time0 = time.time()
    for _ in range(1):
        jax.block_until_ready(forward(h, x, v))
    time1 = time.time()

    print((time1 - time0) / (30 * 1))

if __name__ == "__main__":
    run()
