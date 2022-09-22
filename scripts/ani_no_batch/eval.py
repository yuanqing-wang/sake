import jax
import jax.numpy as jnp
import numpy as onp

N_DEVICES = 1
class Collater(object):
    def __init__(self, ds_tr, batch_size=128, n_devices=N_DEVICES):
        self.ds_tr = ds_tr
        self.i, self.x, self.y = self.ds_tr
        self.batch_size = batch_size
        self.n_devices = n_devices

    def get_statistics(self):
        return self.y.mean(), self.y.std()
    
    def __len__(self):
        return len(self.y)

    def _shuffle(self):
        n_data = len(self)
        n_batches = int(len(self) / (self.batch_size * self.n_devices))
        idxs = onp.random.permutation(n_data)[:n_batches * self.batch_size * self.n_devices]
        idxs = idxs.reshape(n_batches, self.n_devices, self.batch_size)
        idxs = list(idxs)
        self.idxs = idxs

    def __iter__(self):
        self._shuffle()
        return self

    @staticmethod
    def make_edge_mask(m):
        return jnp.expand_dims(m, -1) * jnp.expand_dims(m, -2)

    @staticmethod
    def sum_mask(m):
        return jnp.sign(m.sum(-1, keepdims=True))

    def __next__(self):
        if len(self.idxs) == 0:
            raise StopIteration
        else:
            idxs = self.idxs.pop()
            i = jnp.array(self.i[idxs])
            x = jnp.array(self.x[idxs])
            y = jnp.array(self.y[idxs])
            m = ( i > 0 )
            m = self.make_edge_mask(m)
            i = jax.nn.one_hot(i, 5)

            return i, x, y, m








def run():
    ds_tr = onp.load("ds_tr.npy", allow_pickle=True)
    mean, std = Collater(ds_tr).get_statistics()
    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=mean, std=std)

    # data = ANIDataset()
    # ds_vl = onp.load("ds_tr.npy", allow_pickle=True)[()]
    ds_vl = onp.load("ds_te.npy", allow_pickle=True)
    collater = Collater(ds_vl)

    import sake
    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=6,
        update=False,
    )

    sum_mask = collater.sum_mask

    def get_y_pred(params, i, x, m):
        y_pred, _, __ = model.apply(params, i, x, mask=m)
        y_pred = y_pred * sum_mask(m)
        y_pred = y_pred.sum(axis=-2)
        y_pred = coloring(y_pred)
        return y_pred

    def get_loss(params, i, x, y, m):
        y_pred = get_y_pred(params, i, x, m)
        print(y_pred)
        loss = jnp.abs(y_pred - y).mean()
        return loss


    from flax.training.checkpoints import restore_checkpoint
    state = restore_checkpoint("_checkpoint", None)
    params = state['params']

    from tqdm import tqdm
    count = 0
    loss = 0.0
    for i, x, y, m in collater:
        print(i.shape)
        _loss = get_loss(params, i, x, y, m).item()
        loss = loss + _loss
        count += i.shape[0]
        print(_loss, flush=True)
    print(loss / float(count))

if __name__ == "__main__":
    run()
