import jax
import jax.numpy as jnp
import optax
import numpy as onp
import sake
import tqdm
from run import get_data
from ase.db import connect


def get_test_data():
    xs_in, hs_in, us_in, fs_in = [], [], [], []
    xs_out, hs_out, us_out, fs_out = [], [], [], []

    with connect("iso17/test_within.db") as conn:
        for row in conn.select():
            xs_in.append(jnp.array(row["positions"]))
            us_in.append(jnp.array(row["total_energy"]))
            h = jax.nn.one_hot(row["numbers"], 10)
            hs_in.append(h)
            f = jnp.array(row.data["atomic_forces"])
            fs_in.append(f)
    xs_in, hs_in, us_in, fs_in = jnp.stack(xs_in), jnp.stack(hs_in), jnp.stack(us_in), jnp.stack(fs_in)
    xs_in, hs_in, us_in, fs_in = xs_in.astype(jnp.float32), hs_in.astype(jnp.float32), us_in.astype(jnp.float32), fs_in.astype(jnp.float32)

    with connect("iso17/test_other.db") as conn:
        for row in conn.select():
            xs_out.append(jnp.array(row["positions"]))
            us_out.append(jnp.array(row["total_energy"]))
            h = jax.nn.one_hot(row["numbers"], 10)
            hs_out.append(h)
            f = jnp.array(row.data["atomic_forces"])
            fs_out.append(f)
    xs_out, hs_out, us_out, fs_out = jnp.stack(xs_out), jnp.stack(hs_out), jnp.stack(us_out), jnp.stack(fs_out)
    xs_out, hs_out, us_out, fs_out = xs_out.astype(jnp.float32), hs_out.astype(jnp.float32), us_out.astype(jnp.float32), fs_out.astype(jnp.float32)

    us_in = jnp.expand_dims(us_in, -1)
    us_out = jnp.expand_dims(us_out, -1)

    return xs_in, hs_in, us_in, fs_in, xs_out, hs_out, us_out, fs_out

def run():
    batch_size = 100
    x_tr, i_tr, e_tr, f_tr, x_vl, i_vl, e_vl, f_vl = get_data()
    x_in, i_in, e_in, f_in, x_out, i_out, e_out, f_out = get_test_data()

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

    from flax.training.checkpoints import restore_checkpoint
    state = restore_checkpoint("__checkpoint", None)
    params = state['params']

    n_vl = len(x_vl)
    n_batches = int(n_vl / batch_size)

    i = i_vl.reshape(n_batches, batch_size, i_vl.shape[-2], i_vl.shape[-1])
    x = x_vl.reshape(n_batches, batch_size, x_vl.shape[-2], x_vl.shape[-1])
    e = e_vl.reshape(n_batches, batch_size, e_vl.shape[-1])
    f = f_vl.reshape(n_batches, batch_size, f_vl.shape[-2], f_vl.shape[-1])

    f_losses = []
    e_losses = []
    for idx in range(n_batches):
        f_losses.append(jnp.abs(get_f_pred(params, i[idx], x[idx]) - f[idx]).mean().item())
        e_losses.append(jnp.abs(get_e_pred(params, i[idx], x[idx]) - e[idx]).mean().item())
    f_losses = onp.mean(f_losses)
    e_losses = onp.mean(e_losses)

    print("validation", f_losses, e_losses)

    n_in = len(x_in)
    n_batches = int(n_in / batch_size)

    i = i_in.reshape(n_batches, batch_size, i_in.shape[-2], i_in.shape[-1])
    x = x_in.reshape(n_batches, batch_size, x_in.shape[-2], x_in.shape[-1])
    e = e_in.reshape(n_batches, batch_size, e_in.shape[-1])
    f = f_in.reshape(n_batches, batch_size, f_in.shape[-2], f_in.shape[-1])

    fs_hat = []
    es_hat = []
    fs = []
    es = []
    for idx in range(n_batches):
        fs_hat.append(get_f_pred(params, i[idx], x[idx]))
        es_hat.append(get_e_pred(params, i[idx], x[idx]))
        fs.append(f[idx])
        es.append(e[idx])
    fs_hat, es_hat, fs, es = 1000.0 * jnp.concatenate(fs_hat), 1000.0 * jnp.concatenate(es_hat), 1000.0 * jnp.concatenate(fs), 1000.0 * jnp.concatenate(es)
    mean, low, high = sake.utils.bootstrap_mae(fs_hat, fs)
    print("in f, %.2f_{%.2f}^{%.2f}" % (mean, low, high))
    mean, low, high = sake.utils.bootstrap_mae(es_hat, es)
    print("in e, %.2f_{%.2f}^{%.2f}" % (mean, low, high))

    n_out = len(x_out)
    n_batches = int(n_out / batch_size)

    i = i_out.reshape(n_batches, batch_size, i_out.shape[-2], i_out.shape[-1])
    x = x_out.reshape(n_batches, batch_size, x_out.shape[-2], x_out.shape[-1])
    e = e_out.reshape(n_batches, batch_size, e_out.shape[-1])
    f = f_out.reshape(n_batches, batch_size, f_out.shape[-2], f_out.shape[-1])

    fs_hat = []
    es_hat = []
    fs = []
    es = []
    for idx in range(n_batches):
        fs_hat.append(get_f_pred(params, i[idx], x[idx]))
        es_hat.append(get_e_pred(params, i[idx], x[idx]))
        fs.append(f[idx])
        es.append(e[idx])
    fs_hat, es_hat, fs, es = 1000.0 * jnp.concatenate(fs_hat), 1000.0 * jnp.concatenate(es_hat), 1000.0 * jnp.concatenate(fs), 1000.0 * jnp.concatenate(es)
    mean, low, high = sake.utils.bootstrap_mae(fs_hat, fs)
    print("out f, %.2f_{%.2f}^{%.2f}" % (mean, low, high))
    mean, low, high = sake.utils.bootstrap_mae(es_hat, es)
    print("out e, %.2f_{%.2f}^{%.2f}" % (mean, low, high))


if __name__ == "__main__":
    run()
