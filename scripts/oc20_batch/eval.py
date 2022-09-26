import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
import tqdm
import random


class Collater(object):
    def __init__(self, ds_tr, batch_size=64):
        self.ds_tr = self._move_to_device(ds_tr)
        self.batch_size = batch_size
        self.pointers = []

    @staticmethod
    def _move_to_device(ds_tr):
        for length in ds_tr:
            ds_tr[length]['i'] = jnp.array(ds_tr[length]['i'])
            ds_tr[length]['x'] = jnp.array(ds_tr[length]['x'])
            ds_tr[length]['y'] = jnp.array(ds_tr[length]['y'])
        return ds_tr

    def get_statistics(self):
        ys = jnp.concatenate([self.ds_tr[length]['y'].reshape(-1) for length in self.ds_tr])
        return ys.mean(), ys.std()

    def get_pointers(self):
        pointers = []
        for length in self.ds_tr:
            n_data = self.ds_tr[length]['x'].shape[0]
            n_batches = int(n_data / self.batch_size)
            idxs = onp.random.permutation(n_data)[:n_batches*self.batch_size]
            idxs = idxs.reshape(n_batches, self.batch_size)
            for idx in idxs:
                pointers.append((length, idx))
        import random
        random.shuffle(pointers)
        self.pointers = pointers
        return pointers

    def get_from_pointer(self, pointer):
        length, idxs = pointer
        return (
                self.ds_tr[length]['i'][idxs],
                jnp.array(self.ds_tr[length]['x'][idxs]),
                jnp.expand_dims(jnp.array(self.ds_tr[length]['y'][idxs]), -1),
        )

    def __iter__(self):
        self.get_pointers()
        return self

    def __next__(self):
        if len(self.pointers) == 0:
            raise StopIteration
        else:
            pointer = self.pointers.pop()
            return self.get_from_pointer(pointer)

    def __len__(self):
        return len(self.get_pointers())

def run(args):
    data = onp.load("is2re10k.npy", allow_pickle=True)
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
        update=[False, False, False, False, True, True],
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

    # @jax.jit
    def epoch(state, data):
        def loop_body(idx, state):
            x, y, i = data[idx]
            i = jax.nn.one_hot(i, i_max)
            state = step(state, i, x, y)
            return state

        # state = jax.lax.fori_loop(0, len(data), loop_body, state)
        idxs = list(range(len(data)))
        random.shuffle(idxs)
        for idx in idxs:
            state = loop_body(idx, state)

        return state

    key = jax.random.PRNGKey(2666)
    x0, _, i0 = next(iter(data))
    i0 = jax.nn.one_hot(i0, i_max)
    params = model.init(key, i0, x0)

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint, restore_checkpoint
    state = restore_checkpoint("_checkpoint", None)
    params = state['params']

    for x, y, i in data:
        loss = loss_fn(params, i, x, y)
        print(loss)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    args = parser.parse_args()
    run(args)
