import jax
import jax.numpy as jnp
import optax
import numpy as onp
import sake
import tqdm
from ase.db import connect

def get_data():
    xs, hs, us, fs = [], [], [], []
    with connect("iso17/reference.db") as conn:
        for row in conn.select():
            xs.append(jnp.array(row["positions"]))
            us.append(jnp.array(row["total_energy"]))
            # h = jnp.zeros((19, 10))
            # h[jnp.arange(19), row["numbers"]] = 1
            h = jax.nn.one_hot(row["numbers"], 10)
            hs.append(h)
            f = jnp.array(row.data["atomic_forces"])
            fs.append(f)
    xs, hs, us, fs = jnp.stack(xs), jnp.stack(hs), jnp.stack(us), jnp.stack(fs)
    xs, hs, us, fs = xs.astype(jnp.float32), hs.astype(jnp.float32), us.astype(jnp.float32), fs.astype(jnp.float32)
    idxs_tr = open("iso17/train_ids.txt").readlines()
    idxs_vl = open("iso17/validation_ids.txt").readlines()
    idxs_tr = jnp.array([int(x.strip())-1 for x in idxs_tr])
    idxs_vl = jnp.array([int(x.strip())-1 for x in idxs_vl])

    xs_tr, hs_tr, us_tr, fs_tr = xs[idxs_tr], hs[idxs_tr], us[idxs_tr], fs[idxs_tr]
    xs_vl, hs_vl, us_vl, fs_vl = xs[idxs_vl], hs[idxs_vl], us[idxs_vl], fs[idxs_vl]
    us_tr = jnp.expand_dims(us_tr, -1)
    us_vl = jnp.expand_dims(us_vl, -1)

    return xs_tr, hs_tr, us_tr, fs_tr, xs_vl, hs_vl, us_vl, fs_vl

def run():
    batch_size = 128
    x_tr, i_tr, e_tr, f_tr, x_vl, i_vl, e_vl, f_vl = get_data()
    n_tr = len(x_tr)
    n_batches = int(n_tr / batch_size)
    print(n_tr, flush=True)

    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=e_tr.mean(), std=e_tr.std())
    e_tr_mean = e_tr.mean().item()

    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=6,
    )

    @jax.jit
    def get_e_pred(params, i, x):
        e_pred, _, __ = model.apply(params, i, x)
        e_pred = e_pred.sum(axis=-2)
        e_pred = coloring(e_pred)
        return e_pred

    def get_e_pred_sum(params, i, x):
        e_pred = get_e_pred(params, i, x)
        return -e_pred.sum()

    get_f_pred = jax.jit(lambda params, i, x: jax.grad(get_e_pred_sum, argnums=(2,))(params, i, x)[0])

    def loss_fn(params, i, x, e, f):
        e_pred = get_e_pred(params, i, x)
        f_pred = get_f_pred(params, i, x)
        e_loss = jnp.abs(e_pred - e).mean()
        f_loss = jnp.abs(f_pred - f).mean()
        return f_loss + e_loss * 0.001

    @jax.jit
    def step(state, i, x, e, f):
        params = state.params
        grads = jax.grad(loss_fn)(params, i, x, e, f)
        state = state.apply_gradients(grads=grads)
        return state

    # @jax.jit
    def epoch(state, i_tr, x_tr, e_tr, f_tr):
        key = jax.random.PRNGKey(state.step)
        idxs = jax.random.permutation(key, jnp.arange(x_tr.shape[0]))
        i_tr, x_tr, e_tr, f_tr = i_tr[idxs], x_tr[idxs], e_tr[idxs], f_tr[idxs]

        i_tr = i_tr.reshape(n_batches, batch_size, i_tr.shape[-2], i_tr.shape[-1])
        x_tr = x_tr.reshape(n_batches, batch_size, x_tr.shape[-2], x_tr.shape[-1])
        e_tr = e_tr.reshape(n_batches, batch_size, e_tr.shape[-1])
        f_tr = f_tr.reshape(n_batches, batch_size, f_tr.shape[-2], f_tr.shape[-1])

        def loop_body(idx_batch, state):
            i = i_tr[idx_batch]
            x = x_tr[idx_batch]
            e = e_tr[idx_batch]
            f = f_tr[idx_batch]
            state = step(state, i, x, e, f)
            return state

        # state = jax.lax.fori_loop(0, n_batches, loop_body, state)
        for idx_batch in range(n_batches):
            state = loop_body(idx_batch, state)

        return state

    from functools import partial

    @partial(jax.jit, static_argnums=(6,))
    def many_epochs(state, i_tr, x_tr, e_tr, f_tr, n=10):
        def loop_body(idx, state):
            state = epoch(state, i_tr, x_tr, e_tr, f_tr)
            return state
        state = jax.lax.fori_loop(0, n, loop_body, state)
        return state

    key = jax.random.PRNGKey(2666)
    x0 = x_tr[:batch_size]
    i0 = i_tr[:batch_size]
    params = model.init(key, i0, x0)
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-3,
        warmup_steps=10 * n_batches,
        decay_steps=90 * n_batches,
    )

    optimizer = optax.chain(
        optax.additive_weight_decay(1e-12),
        optax.clip(1.0),
        optax.adam(learning_rate=scheduler),
    )

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    for idx_batch in tqdm.tqdm(range(100)):
        import time
        state = epoch(state, i_tr, x_tr, e_tr, f_tr)
        save_checkpoint("__checkpoint", target=state, step=idx_batch)

if __name__ == "__main__":
    import sys
    run()
