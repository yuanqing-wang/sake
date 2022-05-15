import jax
import jax.numpy as jnp
import numpy as onp
from run import Collator

def run():
    # data = ANIDataset()
    ds_vl = onp.load("ds_vl.npy", allow_pickle=True)[()]
    collater = Collater(ds_vl)

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
        loss = jnp.abs(y_pred - y).sum()
        return loss

    from flax.training.checkpoints import restore_checkpoint
    state = restore_checkpoint("_checkpoint", None)

    from tqdm import tqdm
    count = 0
    loss = 0.0
    for i, x, y in collater:
        loss += get_loss(state.params, i, x, y).item()
        count += i.shape[0]
    print(loss / float(count))

if __name__ == "__main__":
    run()
