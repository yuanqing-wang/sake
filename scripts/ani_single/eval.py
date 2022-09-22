import jax
import jax.numpy as jnp
import numpy as onp
from run import Collater

def run():
    ds_tr = onp.load("ds_tr.npy", allow_pickle=True)[()]
    mean, std = Collater(ds_tr).get_statistics()
    from sake.utils import coloring
    from functools import partial
    print(mean, std)
    coloring = partial(coloring, mean=mean, std=std)

    # data = ANIDataset()
    # ds_vl = onp.load("ds_tr.npy", allow_pickle=True)[()]
    ds_vl = onp.load("ds_te.npy", allow_pickle=True)[()]
    collater = Collater(ds_vl)

    import sake
    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=6,
    )

    def get_y_pred(params, i, x):
        y_pred, _, __ = model.apply(params, i, x)
        y_pred = y_pred.sum(axis=-2)
        y_pred = coloring(y_pred)
        return y_pred

    def get_loss(params, i, x, y):
        y_pred = get_y_pred(params,i, x)
        loss = jnp.abs(y_pred - y).sum()
        return loss

    from flax.training.checkpoints import restore_checkpoint
    state = restore_checkpoint("_checkpoint", None)
    from flax.jax_utils import unreplicate
    params = unreplicate(state['params'])

    from tqdm import tqdm
    count = 0
    loss = 0.0
    for i, x, y in collater:
        _loss = get_loss(params, i, x, y).item()
        loss = loss + _loss
        count += i.shape[0]
        print(_loss)
    print(loss / float(count))

if __name__ == "__main__":
    run()
