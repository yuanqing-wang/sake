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

    x = x_tr[:100]
    q = q_tr[:100]
    v = v_tr[:100]

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

    params = model.init(jax.random.PRNGKey(2666), h, x, v)

    @jax.jit
    def forward(h, x, v):
        h, x, v = model.apply(params, h, x, v)
        return h, x, v

    for _ in range(5):
        jax.block_until_ready(forward(h, x, v))


    import time
    time0 = time.time()
    for _ in range(10):
        jax.block_until_ready(forward(h, x, v))
    time1 = time.time()

    print(0.1 * (time1 - time0))

if __name__ == "__main__":
    run()
