import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
import tqdm
import random


BATCH_SIZE = 4

class Collater(object):
    def __init__(self, ds_tr, batch_size=BATCH_SIZE, n_devices=8):
        self.ds_tr = ds_tr # self._move_to_device(ds_tr)
        self.batch_size = batch_size
        self.pointers = []
        self.n_devices = n_devices

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

    
    def __len__(self):
        return len(self.get_pointers())

    def get_pointers(self):
        pointers = []
        for length in self.ds_tr:
            n_data = self.ds_tr[length]['x'].shape[0]
            n_batches = int(n_data / (self.batch_size * self.n_devices))
            idxs = onp.random.permutation(n_data)[:n_batches*self.batch_size*self.n_devices]
            idxs = idxs.reshape(n_batches, self.batch_size*self.n_devices)
            for idx in idxs:
                pointers.append((length, idx))
        import random
        random.shuffle(pointers)
        self.pointers = pointers
        return pointers

    def get_from_pointer(self, pointer):
        length, idxs = pointer
        i = jax.nn.one_hot(jnp.array(self.ds_tr[length]['i'][idxs]), 4)
        x = jnp.array(self.ds_tr[length]['x'][idxs])
        y = jnp.expand_dims(jnp.array(self.ds_tr[length]['y'][idxs]), -1)

        i = i.reshape(self.n_devices, self.batch_size, *i.shape[1:])
        x = x.reshape(self.n_devices, self.batch_size, *x.shape[1:])
        y = y.reshape(self.n_devices, self.batch_size, *y.shape[1:])

        return i, x, y

    def __iter__(self):
        self.get_pointers()
        return self

    def __next__(self):
        if len(self.pointers) == 0:
            raise StopIteration
        else:
            pointer = self.pointers.pop()
            return self.get_from_pointer(pointer)

def run(args):
    data = onp.load("is2re_all.npy", allow_pickle=True)[()]
    collater = Collater(data)
    print(len(collater))
    i_max = 0
    counter = 0
    for i, x, y in collater:
        print(i.shape, x.shape)
        i_max = max(i.max(), i_max)
        counter += 1
    i_max = i_max + 1
    from sake.utils import coloring
    from functools import partial
    y_mean, y_std = collater.get_statistics()
    coloring = partial(coloring, mean=y_mean, std=y_std)

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
    x0, _, i0 = next(iter(collater))
    i0 = jax.nn.one_hot(i0, i_max)
    print(i0.shape, x0.shape)
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
