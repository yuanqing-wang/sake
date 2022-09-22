import jax
import jax.numpy as jnp
import numpy as onp
import flax
from functools import partial

BATCH_SIZE = 128

class Collater(object):
    def __init__(self, ds_tr, batch_size=BATCH_SIZE):
        self.ds_tr = ds_tr
        self.i, self.x, self.y = self.ds_tr
        self.batch_size = batch_size

    def get_statistics(self):
        y = self.y
        y = onp.concatenate(y)
        return y.mean(), y.std()
    
    def __len__(self):
        return len(self.y)

    def _shuffle(self):
        n_data = len(self)
        idxs = onp.random.permutation(n_data)
        idxs = list(idxs)
        self.idxs = idxs

    def __iter__(self):
        self._shuffle()
        return self

    def __next__(self):
        if len(self.idxs) == 0:
            raise StopIteration
        else:
            idx = self.idxs.pop()
            i, x, y = self.i[idx], self.x[idx], self.y[idx]
            y = onp.expand_dims(y, -1)
            length = len(y)
            idxs = onp.random.randint(low=0, high=length, size=self.batch_size)
            i, x, y = i[idxs], x[idxs], y[idxs]
            i = jax.nn.one_hot(i, 4)
            x = jnp.array(x)
            y = jnp.array(y)
            return i, x, y

def run():
    # data = ANIDataset()
    ds_tr = onp.load("ds_tr.npy", allow_pickle=True)
    collater = Collater(ds_tr)

    mean, std = collater.get_statistics()
    import sake
    from functools import partial
    coloring = partial(sake.utils.coloring, mean=mean, std=std)

    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=6,
        update=[False, False, False, False, True, True],
    )
    i, x, y = next(iter(collater))
    params = model.init(jax.random.PRNGKey(2666), i, x)


    def get_y_pred(params, i, x):
        y_pred, _, __ = model.apply(params, i, x)
        y_pred = y_pred.sum(axis=-2)
        y_pred = coloring(y_pred)
        return y_pred

    def get_loss(params, i, x, y):
        y_pred = get_y_pred(params, i, x)
        loss = jnp.abs(y_pred - y).mean()
        return loss

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint, restore_checkpoint
    # state = restore_checkpoint("_checkpoint", target=None)
    # params = state["params"]
    # idx_batch0 = int(state["step"])
    idx_batch0 = 0

    import optax
    n_batches = len(collater)

    optimizer = optax.chain(
        optax.adam(1e-4),
    )
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    @jax.jit
    def step(state, i, x, y):
        params = state.params
        value, grads = jax.value_and_grad(get_loss)(params, i, x, y)
        state = state.apply_gradients(grads=grads)
        return state, value

    for idx_batch in range(idx_batch0, 10000000):
        for i, x, y in collater:
            state, value = step(state, i, x, y)
        save_checkpoint("_checkpoint", target=state, step=idx_batch, keep_every_n_steps=1)

if __name__ == "__main__":
    run()
