import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
import tqdm

def run():
    ds_tr, ds_vl, ds_te = onp.load("train.npz"), onp.load("valid.npz"), onp.load("test.npz")
    i_tr, i_vl, i_te = ds_tr["charges"], ds_vl["charges"], ds_te["charges"]
    x_tr, x_vl, x_te = ds_tr["positions"], ds_vl["positions"], ds_te["positions"]
    
    m_tr, m_vl, m_te = (i_tr > 0).sum(-1) == 19, (i_vl > 0).sum(-1) == 19, (i_te > 0).sum(-1) == 19

    i_tr, i_vl, i_te = i_tr[m_tr, :19], i_vl[m_vl, :19], i_te[m_te, :19]
    x_tr, x_vl, x_te = x_tr[m_tr, :19, :], x_vl[m_vl, :19, :], x_te[m_te, :19, :]
    i_tr, i_vl, i_te = jax.nn.one_hot(i_tr, i_tr.max()+1), jax.nn.one_hot(i_vl, i_tr.max()+1), jax.nn.one_hot(i_te, i_tr.max()+1)

    BATCH_SIZE = 16
    N_BATCHES = len(i_tr) // BATCH_SIZE

    model = sake.flows.AugmentedFlowModel(depth=3, mp_depth=3)
    key = jax.random.PRNGKey(2666)
    h = jnp.zeros((BATCH_SIZE, 19, 10))
    
    print(x_tr.shape, i_tr.shape)
    params = model.init(key, h, x_tr[:BATCH_SIZE], x_tr[:BATCH_SIZE])
    prior = sake.flows.CenteredGaussian

    # @jax.jit
    def get_loss(params, key, i, x):
        v = prior.sample(key=key, shape=x.shape)
        x, v, sum_log_det = model.apply(params, i, x, v, method=model.f_backward)
        loss = (-prior.log_prob(x) - prior.log_prob(v) + sum_log_det).mean()
        return loss

    @jax.jit
    def step(state, key, i, x):
        key, subkey = jax.random.split(key)
        params = state.params
        grads = jax.grad(get_loss)(params, subkey, i, x)
        state = state.apply_gradients(grads=grads)
        return state, key

    @jax.jit
    def epoch(state, key, _i, _x):
        def fn(idx, _state):
            state, key = _state
            state, key = step(state, key, _i[idx], _x[idx])
            _state = state, key
            return _state

        state, key = jax.lax.fori_loop(0, N_BATCHES, fn, (state, key))
        return state, key

    import optax
    optimizer = optax.chain(
        optax.additive_weight_decay(1e-5),
        optax.clip(1.0),
        optax.adam(1e-5),
    )
    optimizer = optax.apply_if_finite(optimizer, 5)

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
         apply_fn=model.apply, params=params, tx=optimizer,
    )

    import tqdm
    for idx_batch in tqdm.tqdm(range(5000)):
        idxs = jax.random.permutation(jax.random.PRNGKey(state.step), len(x_tr))[:BATCH_SIZE*N_BATCHES]
        _x = x_tr[idxs].reshape(N_BATCHES, BATCH_SIZE, 19, 3)
        _i = i_tr[idxs].reshape(N_BATCHES, BATCH_SIZE, 19, 10)
        state, key = epoch(state, key, _i, _x)
        assert state.opt_state.notfinite_count <= 10
        save_checkpoint("__checkpoint", target=state, step=idx_batch)

if __name__ == "__main__":
    import sys
    run()
