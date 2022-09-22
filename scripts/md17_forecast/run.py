import jax
import jax.numpy as jnp
import optax
import sake

def run():
    from data.dataset import MD17Dataset
    ds_tr = MD17Dataset("train", 500, 5000, "data", "aspirin")
    ds_vl = MD17Dataset("val", 2000, 5000, "data", "aspirin")
    ds_te = MD17Dataset("test", 2000, 5000, "data", "aspirin")

    x0_tr, v0_tr, x1_tr = jnp.array(ds_tr.x_0), jnp.array(ds_tr.v_0), jnp.array(ds_tr.x_t)
    x0_vl, v0_vl, x1_vl = jnp.array(ds_vl.x_0), jnp.array(ds_vl.v_0), jnp.array(ds_vl.x_t)
    x0_te, v0_te, x1_te = jnp.array(ds_te.x_0), jnp.array(ds_te.v_0), jnp.array(ds_te.x_t)
    h = jnp.array(ds_tr.mole_idx, jnp.int32)
    h = jax.nn.one_hot(h, 8)

    def make_h(v0):
        v0_norm = jnp.linalg.norm(v0, axis=-1, keepdims=True)
        eye = h
        # eye = jnp.eye(v0.shape[-2])
        # eye = jnp.concatenate([eye, h], axis=-1)
        eye = jnp.expand_dims(eye, 0)
        eye = jnp.broadcast_to(eye, (*v0_norm.shape[:-1], eye.shape[-1]))
        h_new = jnp.concatenate([v0_norm, eye], axis=-1)
        return h_new

    h_tr, h_vl, h_te = make_h(v0_tr), make_h(v0_vl), make_h(v0_te)

    model = sake.models.DenseSAKEModel(
        hidden_features=16,
        out_features=1,
        depth=4,
    )

    params = model.init(jax.random.PRNGKey(2666), h_tr, x0_tr, v=v0_tr)

    def loss(params, h, x0, x1, v0):
        _, x1_hat, __ = model.apply(params, h, x0, v=v0)
        loss = ((x1_hat - x1) ** 2).mean()
        return loss

    # @jax.jit
    def step(state, h, x0, x1, v0):
        grads = jax.grad(loss)(state.params, h, x0, x1, v0)
        state = state.apply_gradients(grads=grads)
        return state

    @jax.jit
    def epoch(state, key):
        idxs = jax.random.permutation(key=key, x=500)
        _h_tr = h_tr[idxs]
        _x0_tr = x0_tr[idxs]
        _x1_tr = x1_tr[idxs]
        _v0_tr = v0_tr[idxs]

        _h_tr = _h_tr.reshape(5, 100, *h_tr.shape[1:])
        _x0_tr = _x0_tr.reshape(5, 100, *x0_tr.shape[1:])
        _x1_tr = _x1_tr.reshape(5, 100, *x1_tr.shape[1:])
        _v0_tr = _v0_tr.reshape(5, 100, *v0_tr.shape[1:])

        for idx in range(5):
            _h = _h_tr[idx]
            _x0 = _x0_tr[idx]
            _x1 = _x1_tr[idx]
            _v0 = _v0_tr[idx]
            state = step(state, _h, _x0, _x1, _v0)
        
        return state

    @jax.jit
    def eval(state):
        _, x1_hat_vl, __ = model.apply(state.params, h_vl, x0_vl, v0_vl)
        _, x1_hat_te, __ = model.apply(state.params, h_te, x0_te, v0_te)
        _, x1_hat_tr, __ = model.apply(state.params, h_tr, x0_tr, v0_tr)
        error_tr = jnp.mean(jnp.abs(x1_hat_tr - x1_tr))
        error_vl = jnp.mean(jnp.abs(x1_hat_vl - x1_vl))
        error_te = jnp.mean(jnp.abs(x1_hat_te - x1_te))
        return error_tr, error_vl, error_te

    optimizer = optax.chain(
        optax.additive_weight_decay(1e-10),
        optax.adam(learning_rate=0.0005),
    )

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    key = jax.random.PRNGKey(2666)
    for idx_epoch in range(10000):
        this_key, key = jax.random.split(key)
        state = epoch(state, this_key)

        if idx_epoch % 100 == 0:
            error_tr, error_vl, error_te = eval(state)
            print(error_tr, error_vl, error_te, flush=True)

if __name__ == "__main__":
    run()
