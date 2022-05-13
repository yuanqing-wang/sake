import jax
import jax.numpy as jnp
import numpy as onp
import flax
from functools import partial

BATCH_SIZE = 1024

class Collater(object):
    def __init__(self, ds_tr, batch_size=BATCH_SIZE, n_device=8):
        self.ds_tr = ds_tr
        self.pointers = []
        self.batch_size = batch_size
        self.n_device = n_device

    def __len__(self):
        return len(self.get_pointers())

    def _move_to_device(ds_tr):
        for length in ds_tr:
            ds_tr[length]['i'] = jnp.array(ds_tr[length]['i'])
            ds_tr[length]['x'] = jnp.array(ds_tr[length]['x'])
            ds_tr[length]['y'] = jnp.array(ds_tr[length]['y'])
        return ds_tr

    def get_pointers(self):
        pointers = []
        for length in self.ds_tr:
            n_data = self.ds_tr[length]['x'].shape[0]
            n_batches = int(n_data / (self.batch_size * self.n_device))
            idxs = onp.random.permutation(n_data)[:n_batches* (self.batch_size * self.n_device)]
            idxs = idxs.reshape(n_batches,  (self.batch_size * self.n_device))
            for idx in idxs:
                pointers.append((length, idx))
        import random
        random.shuffle(pointers)
        self.pointers = pointers
        return pointers

    def get_from_pointer(self, pointer):
        length, idxs = pointer
        i = jax.nn.one_hot(self.ds_tr[length]['i'][idxs], 4)
        x = jnp.array(self.ds_tr[length]['x'][idxs])
        y = jnp.expand_dims(jnp.array(self.ds_tr[length]['y'][idxs]), -1)

        i = i.reshape(self.n_device, self.batch_size, *i.shape[1:])
        x = x.reshape(self.n_device, self.batch_size, *x.shape[1:])
        y = y.reshape(self.n_device, self.batch_size, *y.shape[1:])

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


def run():
    # data = ANIDataset()
    ds_tr = onp.load("ds_tr.npy", allow_pickle=True)[()]
    collater = Collater(ds_tr)

    import sake
    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=8,
    )

    def get_y_pred(params, i, x):
        y_pred, _, __ = model.apply(params, i, x)
        y_pred = y_pred.sum(axis=-2)
        return y_pred

    def get_loss(params, i, x, y):
        y_pred = get_y_pred(params,i, x)
        loss = jnp.abs(y_pred - y).mean()
        return loss

    i, x, y = next(iter(collater))
    params = model.init(jax.random.PRNGKey(2666), i, x)

    import optax
    n_batches = len(collater)
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-3,
        warmup_steps=100 * n_batches,
        decay_steps=900 * n_batches,
    )

    optimizer = optax.chain(
        optax.additive_weight_decay(1e-5),
        optax.clip(1.0),
        optax.adam(1e-5),
    )

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint

    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    state = flax.jax_utils.replicate(state)

    @partial(jax.pmap, axis_name="batch")
    def step(state, i, x, y):
        params = state.params
        loss, grads = jax.value_and_grad(get_loss)(params, i, x, y)
        loss = jax.lax.pmean(loss, "batch")
        grads = jax.lax.pmean(grads, "batch")
        state = state.apply_gradients(grads=grads)
        return loss, state

    from tqdm import tqdm
    for idx_batch in tqdm(range(50)):
        for i, x, y in collater:
            loss, state = step(state, i, x, y)
            print(loss)
        save_checkpoint("_checkpoint", target=state, step=idx_batch)

if __name__ == "__main__":
    run()
