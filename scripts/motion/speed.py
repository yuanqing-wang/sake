import jax
import jax.numpy as jnp
import optax
import sake

def run():
    from data.dataset import MotionDataset
    ds_tr = MotionDataset("train", 200, 30, "data")
    ds_vl = MotionDataset("val", 600, 30, "data")
    ds_te = MotionDataset("test", 600, 30, "data")

    x0_tr, v0_tr, x1_tr = jnp.array(ds_tr.x_0), jnp.array(ds_tr.v_0), jnp.array(ds_tr.x_t)
    x0_vl, v0_vl, x1_vl = jnp.array(ds_vl.x_0), jnp.array(ds_vl.v_0), jnp.array(ds_vl.x_t)
    x0_te, v0_te, x1_te = jnp.array(ds_te.x_0), jnp.array(ds_te.v_0), jnp.array(ds_te.x_t)

    def make_h(v0):
        v0_norm = jnp.linalg.norm(v0, axis=-1, keepdims=True)
        eye = jnp.eye(v0.shape[-2])
        eye = jnp.expand_dims(eye, 0)
        eye = jnp.broadcast_to(eye, (*v0_norm.shape[:-1], eye.shape[-1]))
        h = jnp.concatenate([v0_norm, eye], axis=-1)
        return h

    h_tr, h_vl, h_te = make_h(v0_tr), make_h(v0_vl), make_h(v0_te)

    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=4,
        update=True,
    )

    params = model.init(jax.random.PRNGKey(2666), h_tr, x0_tr)


    def loss(params):
        _, x1_hat, __ = model.apply(params, h_tr, x0_tr)
        loss = ((x1_hat - x1_tr) ** 2).mean()
        return loss

    # @jax.jit
    def step(state):
        grads = jax.grad(loss)(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    @jax.jit
    def eval(state):
        _, x1_hat_vl, __ = model.apply(state.params, h_vl, x0_vl)
        _, x1_hat_te, __ = model.apply(state.params, h_te, x0_te)
        error_vl = jnp.mean(jnp.abs(x1_hat_vl - x1_vl))
        error_te = jnp.mean(jnp.abs(x1_hat_te - x1_te))
        return error_vl, error_te

    optimizer = optax.chain(
        optax.additive_weight_decay(1e-10),
        optax.adam(learning_rate=0.0005),
    )

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    state = step(state)

    import time
    time0 = time.time()
    for idx_epoch in range(100):
        state = step(state)
    time1 = time.time()
    print((time1 - time0) * 0.01)

if __name__ == "__main__":
    run()
