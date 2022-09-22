import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
import tqdm
import random

def run(target):
    ds_tr = onp.load("train.npz")
    i_tr = ds_tr["charges"]
    x_tr = ds_tr["positions"]
    y_tr = ds_tr[target]
    N_MAX = i_tr.max() + 1
    
    if target + "_thermo" in ds_tr:
        y_tr = y_tr - ds_tr[target + "_thermo"]

    def mask(i, x):
        m = (i > 0)
        x = x[m]
        return x

    y_tr = onp.expand_dims(y_tr, -1)
    x_tr, i_tr, y_tr = list(x_tr), list(i_tr), list(y_tr)
    x_tr = [mask(i, x) for i, x in zip(i_tr, x_tr)]
    i_tr = [mask(i, i) for i in i_tr]

    x_tr = [jnp.array(x) for x in x_tr]
    i_tr = [jax.nn.one_hot(i, N_MAX) for i in i_tr]
    y_tr = [jnp.array(y) for y in y_tr]

    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=jnp.array(y_tr).mean(), std=jnp.array(y_tr).std())

    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=6,
        update=False, # [False, False, False, True, True, True],
    )

    def get_y_hat(params, i, x):
        y_hat, _, __ = model.apply(params, i, x)
        y_hat = y_hat.sum(-2)
        y_hat = coloring(y_hat)
        return y_hat

    def loss_fn(params, i, x, y):
        y_hat = get_y_hat(params, i, x)
        loss = jnp.abs(y - y_hat).mean()
        return loss

    @jax.jit
    def step(state, i, x, y):
        params = state.params
        grads = jax.grad(loss_fn)(params, i, x, y)
        state = state.apply_gradients(grads=grads)
        return state
    
    def epoch(state, i_tr, x_tr, y_tr):
        idxs = list(range(len(i_tr)))
        random.shuffle(idxs)
        def fn(idx, state):
            _i_tr, _x_tr, _y_tr = i_tr[idx], x_tr[idx], y_tr[idx]
            state = step(state, _i_tr, _x_tr, _y_tr)
            return state
        for idx in idxs:
            state = fn(idx, state)
        return state

    key = jax.random.PRNGKey(2666)
    params = model.init(key, i_tr[0], x_tr[0])

    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-4,
        warmup_steps=100 * len(i_tr),
        decay_steps=1900 * len(i_tr),
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

    for idx_batch in tqdm.tqdm(range(2000)):
        state = epoch(state, i_tr, x_tr, y_tr)
        save_checkpoint("_" + target, target=state, step=idx_batch)

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
