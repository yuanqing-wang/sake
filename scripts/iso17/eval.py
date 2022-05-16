import jax
import jax.numpy as jnp
import optax
import numpy as onp
import sake
import tqdm
from run import get_data

def run():
    batch_size = 100
    x_tr, i_tr, e_tr, f_tr, x_vl, i_vl, e_vl, f_vl = get_data()
    n_vl = len(x_vl)
    n_batches = int(n_vl / batch_size)

    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=e_tr.mean(), std=e_tr.std())
    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=6,
    )

    @jax.jit
    def get_e_pred(params, i, x):
        e_pred, _, __ = model.apply(params, i, x)
        e_pred = e_pred.sum(axis=-2)
        e_pred = coloring(e_pred)
        return e_pred

    def get_e_pred_sum(params, i, x):
        e_pred = get_e_pred(params, i, x)
        return -e_pred.sum()

    get_f_pred = jax.jit(lambda params, i, x: jax.grad(get_e_pred_sum, argnums=(2,))(params, i, x)[0])

    i_vl = i_vl.reshape(n_batches, batch_size, i_vl.shape[-2], i_vl.shape[-1])
    x_vl = x_vl.reshape(n_batches, batch_size, x_vl.shape[-2], x_vl.shape[-1])
    e_vl = e_vl.reshape(n_batches, batch_size, e_vl.shape[-1])
    f_vl = f_vl.reshape(n_batches, batch_size, f_vl.shape[-2], f_vl.shape[-1])


    from flax.training.checkpoints import restore_checkpoint
    state = restore_checkpoint("__checkpoint", None)
    params = state['params']

    f_losses = []
    e_losses = []
    for idx in range(n_batches):
        f_losses.append(jnp.abs(get_f_pred(params, i_vl[idx], x_vl[idx]) - f_vl[idx]).mean().item())
        e_losses.append(jnp.abs(get_e_pred(params, i_vl[idx], x_vl[idx]) - e_vl[idx]).mean().item())
    f_losses = onp.mean(f_losses)
    e_losses = onp.mean(e_losses)

    print(f_losses, e_losses)

if __name__ == "__main__":
    run()
