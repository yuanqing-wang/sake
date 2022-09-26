import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
import tqdm
import random

def run(args):
    data = onp.load("is2re_all.npy", allow_pickle=True)
    y_tr = jnp.array([_data[1] for _data in data])
    i_max = max([max(_data[2]) for _data in data]) + 1
    data = [[jnp.array(_data[0]), jnp.array(_data[1]), jnp.array(_data[2])] for _data in data]
    random.shuffle(data)
    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=y_tr.mean(), std=y_tr.std())

    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=64,
        depth=6,
    )

    def get_y_hat(params, i, x):
        y_hat, _, __ = model.apply(params, i, x)
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

    key = jax.random.PRNGKey(2666)
    x0, _, i0 = next(iter(data))
    i0 = jax.nn.one_hot(i0, i_max)
    params = model.init(key, i0, x0)

    optimizer = optax.chain(
        optax.additive_weight_decay(args.weight_decay),
        optax.clip(1.0),
        optax.adam(learning_rate=1e-3),
    )
    optimizer = optax.apply_if_finite(optimizer, 5)

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )
    ds_tr = onp.load("ds_tr.npy", allow_pickle=True)[()]
    collater = Collater(ds_tr)

    
    for idx_batch in tqdm.tqdm(range(200)):
        for i, x, y in collater:
            state = step(state, i, x, y)
        assert state.opt_state.notfinite_count <= 10
        save_checkpoint("__checkpoint", target=state, step=idx_batch)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    args = parser.parse_args()
    run(args)
