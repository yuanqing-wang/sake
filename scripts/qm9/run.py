import jax
import jax.numpy as jnp
import optax
import numpy as onp
import sake
import tqdm

def run(target):
    ds_tr, ds_vl, ds_te = onp.load("train.npz"), onp.load("valid.npz"), onp.load("test.npz")
    i_tr, i_vl, i_te = ds_tr["charges"], ds_vl["charges"], ds_te["charges"]
    x_tr, x_vl, x_te = ds_tr["positions"], ds_vl["positions"], ds_te["positions"]
    y_tr, y_vl, y_te = ds_tr[target], ds_vl[target], ds_te[target]
    y_tr, y_vl, y_te = onp.expand_dims(y_tr, -1), onp.expand_dims(y_vl, -1), onp.expand_dims(y_te, -1)
    m_tr, m_vl, m_te = (i_tr > 0), (i_vl > 0), (i_te > 0)

    def make_edge_mask(m):
        return jnp.expand_dims(m, -1) * jnp.expand_dims(m, -2)

    def sum_mask(m):
        return jnp.sign(m.sum(-1, keepdims=True))

    for _var in ["i", "x", "y", "m"]:
        for _split in ["tr", "vl", "te"]:
            locals()["%s_%s" % (_var, _split)] = jnp.array(locals()["%s_%s" % (_var, _split)])

    m_tr, m_vl, m_te = make_edge_mask(m_tr), make_edge_mask(m_vl), make_edge_mask(m_te)

    BATCH_SIZE = 128
    N_BATCHES = len(i_tr) // batch_size

    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=y_tr.mean(), std=y_tr.std())

    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=8,
    )

    def get_y_hat(params, i, x, m):
        y_hat = model.apply(params, i, x, mask=m)
        y_hat = coloring(y_hat)
        y_hat = y_hat * sum_mask(m)
        y_hat = y_hat.sum(-2)
        return y_hat

    def loss_fn(params, i, x, m, y):
        y_hat = get_y_hat(params, i, x, mask=m)
        loss = ((y - y_hat) ** 2).mean()
        return loss

    def step(state, i, x, m, y):
        params = state.params
        grads = jax.grad(loss_fn)(params, i, x, m, y)
        state = state.apply_gradients(grads=grads)
        return state

    def epoch(state, i_tr, x_tr, m_tr, y_tr):
        key = jax.random.PRNGKey(state.step)
        idxs = jax.random.permutation(key, jnp.arange(BATCH_SIZE * N_BATCHES))
        i_tr = i_tr[:BATCH_SIZE * N_BATCHES].reshape(N_BATCHES, BATCH_SIZE, *i_tr.shape[2:])
        x_tr = x_tr[:BATCH_SIZE * N_BATCHES].reshape(N_BATCHES, BATCH_SIZE, *x_tr.shape[2:])
        m_tr = x_tr[:BATCH_SIZE * N_BATCHES].reshape(N_BATCHES, BATCH_SIZE, *m_tr.shape[2:])
        y_tr = y_tr[:BATCH_SIZE * N_BATCHES].reshape(N_BATCHES, BATCH_SIZE, 1)

        def loop_body(idx_batch, state):
            i = i_tr[idx_batch]
            x = i_tr[idx_batch]
            m = m_tr[idx_batch]
            y = y_tr[idx_batch]
            state = step(state, i, x, m, y)
            return state

        state = jax.lax.fori_loop(0, N_BATCHES, loop_body, state)
        return state

    @partial(jax.jit, static_argnums=(5))
    def many_epochs(state, i_tr, x_tr, m_tr, y_tr, n=10):
        def loop_body(state, i_tr, x_tr, m_tr, y_tr):
            state = epoch(state, i_tr, x_tr, m_tr, y_tr)
            return state
        state = jax.lax.fori_loop(0, n, loop_body, state)
        return state

    key = jax.random.PRNGKey(2666)
    i0 = i_tr[:BATCH_SIZE]
    x0 = x_tr[:BATCH_SIZE]
    m0 = m_tr[:BATCH_SIZE]
    y0 = y_tr[:BATCH_SIZE]
    params = model.init(key, i0, x0, m0, y0)
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-3,
        warmup_steps=500 * n_batches,
        decay_steps=4500 * n_batches,
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

    for idx_batch in tqdm.tqdm(range(500)):
        import time
        state = many_epochs(state, i_tr, x_tr, m_tr, y_tr)
        save_checkpoint("_" + target, target=state, step=idx_batch)
