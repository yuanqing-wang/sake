import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
import tqdm

def run(args):
    target = args.target
    ds_tr, ds_vl, ds_te = onp.load("train.npz"), onp.load("valid.npz"), onp.load("test.npz")
    i_tr, i_vl, i_te = ds_tr["charges"], ds_vl["charges"], ds_te["charges"]
    x_tr, x_vl, x_te = ds_tr["positions"], ds_vl["positions"], ds_te["positions"]
    y_tr, y_vl, y_te = ds_tr[target], ds_vl[target], ds_te[target]
    
    if target + "_thermo" in ds_tr:
        y_tr = y_tr - ds_tr[target + "_thermo"]
        y_vl = y_vl - ds_vl[target + "_thermo"]
        y_te = y_te - ds_te[target + "_thermo"]

    y_tr, y_vl, y_te = onp.expand_dims(y_tr, -1), onp.expand_dims(y_vl, -1), onp.expand_dims(y_te, -1)
    m_tr, m_vl, m_te = (i_tr > 0), (i_vl > 0), (i_te > 0)

    def make_edge_mask(m):
        return jnp.expand_dims(m, -1) * jnp.expand_dims(m, -2)

    def sum_mask(m):
        return jnp.sign(m.sum(-1, keepdims=True))

    for _var in ["i", "x", "y", "m"]:
        for _split in ["tr", "vl", "te"]:
            locals()["%s_%s" % (_var, _split)] = jnp.array(locals()["%s_%s" % (_var, _split)])


    i_tr, i_vl, i_te = jax.nn.one_hot(i_tr, i_tr.max()+1), jax.nn.one_hot(i_vl, i_tr.max()+1), jax.nn.one_hot(i_te, i_tr.max()+1)
    m_tr, m_vl, m_te = make_edge_mask(m_tr), make_edge_mask(m_vl), make_edge_mask(m_te)

    BATCH_SIZE = args.batch_size
    N_BATCHES = len(i_tr) // BATCH_SIZE

    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=y_tr.mean(), std=y_tr.std())

    print(y_tr.mean(), y_tr.std())

    class Model(nn.Module):
        def setup(self):
            self.model = sake.models.DenseSAKEModel(
                hidden_features=64,
                out_features=64,
                depth=6,
                update=[False, False, False, False, True, True],
            )

        def __call__(self, i, x, m):
            y, _, __ = self.model(i, x, mask=m)
            y = y * sum_mask(m)
            y = y.sum(-2)
            return y

    model = Model()

    def get_y_hat(params, i, x, m):
        y_hat = model.apply(params, i, x, m=m)
        y_hat = coloring(y_hat)
        return y_hat

    def loss_fn(params, i, x, m, y):
        y_hat = get_y_hat(params, i, x, m)
        loss = jnp.abs(y - y_hat).mean()
        return loss

    def step(state, i, x, m, y):
        params = state.params
        grads = jax.grad(loss_fn)(params, i, x, m, y)
        state = state.apply_gradients(grads=grads)
        return state
    
    @jax.jit
    def step_with_loss(state, i, x, m, y):
        params = state.params
        grads = jax.grad(loss_fn)(params, i, x, m, y)
        state = state.apply_gradients(grads=grads)
        return state
    
    @jax.jit
    def epoch(state, i_tr, x_tr, m_tr, y_tr):
        key = jax.random.PRNGKey(state.step)
        idxs = jax.random.permutation(key, jnp.arange(BATCH_SIZE * N_BATCHES))
        _i_tr = i_tr[idxs][:BATCH_SIZE * N_BATCHES].reshape(N_BATCHES, BATCH_SIZE, *i_tr.shape[1:])
        _x_tr = x_tr[idxs][:BATCH_SIZE * N_BATCHES].reshape(N_BATCHES, BATCH_SIZE, *x_tr.shape[1:])
        _m_tr = m_tr[idxs][:BATCH_SIZE * N_BATCHES].reshape(N_BATCHES, BATCH_SIZE, *m_tr.shape[1:])
        _y_tr = y_tr[idxs][:BATCH_SIZE * N_BATCHES].reshape(N_BATCHES, BATCH_SIZE, 1)

        def loop_body(idx, state):
            # i, x, m, y = next(iterator)
            # i, x, m, y = jnp.squeeze(i), jnp.squeeze(x), jnp.squeeze(m), jnp.squeeze(y)
            #
            i, x, m, y = _i_tr[idx], _x_tr[idx], _m_tr[idx], _y_tr[idx]
            state = step_with_loss(state, i, x, m, y)
            return state

        state = jax.lax.fori_loop(0, N_BATCHES, loop_body, state)

        '''
        for i, x, m, y in iterator: 
            state = loop_body(i, x, m, y, state)
        '''

        return state

    @partial(jax.jit, static_argnums=(5))
    def many_epochs(state, i_tr, x_tr, m_tr, y_tr, n=10):
        def loop_body(idx_batch, state):
            state = epoch(state, i_tr, x_tr, m_tr, y_tr)
            return state
        state = jax.lax.fori_loop(0, n, loop_body, state)
        return state

    key = jax.random.PRNGKey(2666)
    i0 = i_tr[:BATCH_SIZE]
    x0 = x_tr[:BATCH_SIZE]
    m0 = m_tr[:BATCH_SIZE]
    y0 = y_tr[:BATCH_SIZE]

    params = model.init(key, i0, x0, m0)

    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=500 * N_BATCHES,
        decay_steps=2000 * N_BATCHES,
    )

    optimizer = optax.chain(
        optax.additive_weight_decay(args.weight_decay),
        optax.clip(1.0),
        optax.adam(learning_rate=scheduler),
    )
    optimizer = optax.apply_if_finite(optimizer, 5)

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    for idx_batch in tqdm.tqdm(range(200)):
        state = many_epochs(state, i_tr, x_tr, m_tr, y_tr)
        assert state.opt_state.notfinite_count <= 10

    def _get_y_hat(inputs):
         x, i = jnp.split(inputs, [3], axis=-1)
         m = make_edge_mask(i.argmax(-1) > 0)
         return get_y_hat(state.params, i, x, m)
    
    inputs = jnp.concatenate([x_tr, i_tr], axis=-1)
    y_tr_hat = jax.lax.map(_get_y_hat, inputs)

    inputs = jnp.concatenate([x_vl, i_vl], axis=-1)
    y_vl_hat = jax.lax.map(_get_y_hat, inputs)

    inputs = jnp.concatenate([x_te, i_te], axis=-1)
    y_te_hat = jax.lax.map(_get_y_hat, inputs)

    print(y_tr_hat)
    
    print("training", sake.utils.bootstrap_mae(y_tr_hat, y_tr))
    print("validation", sake.utils.bootstrap_mae(y_vl_hat, y_vl))
    print("test", sake.utils.bootstrap_mae(y_te_hat, y_te))




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="U")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    args = parser.parse_args()
    run(args)
