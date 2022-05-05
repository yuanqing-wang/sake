import jax
import jax.numpy as jnp
import optax
import numpy as onp
import sake

def run(data_name):
    data = onp.load("%s_dft.npz" % data_name)
    onp.random.seed(2666)
    idxs = onp.random.permutation(len(data['R']))

    x = jnp.array(data['R'][idxs])
    e = jnp.array(data['E'][idxs])
    i = jnp.array(data['z'])
    f = jnp.array(data['F'][idxs])

    i = jnp.expand_dims(jax.nn.one_hot(i, i.max()), 0)

    batch_size = 4
    n_tr = n_vl = 1000
    x_tr = x[:n_tr]
    e_tr = e[:n_tr]
    f_tr = f[:n_tr]

    x_vl = x[n_tr:n_tr+n_vl]
    e_vl = e[n_tr:n_tr+n_vl]
    f_vl = f[n_tr:n_tr+n_vl]

    x_te = x[n_tr+n_vl:]
    e_te = e[n_tr+n_vl:]
    f_te = f[n_tr+n_vl:]

    n_batches = int(n_tr / batch_size)

    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=e_tr.mean(), std=e_tr.std())
    e_tr_mean = e_tr.mean().item()

    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=8,
    )

    @jax.jit
    def get_e_pred(params, x):
        i_tr = jnp.repeat(i, x.shape[0], 0)
        e_pred, _, __ = model.apply(params, i_tr, x)
        e_pred = e_pred.sum(axis=1)
        e_pred = coloring(e_pred)
        return e_pred

    def get_e_pred_sum(params, x):
        e_pred = get_e_pred(params, x)
        return -e_pred.sum()

    get_f_pred = jax.jit(lambda params, x: jax.grad(get_e_pred_sum, argnums=(1,))(params, x)[0])

    from flax.training.checkpoints import restore_checkpoint
    state = restore_checkpoint(data_name, None)
    params = state['params']

    e_vl_hat = get_e_pred(params, x_vl)
    f_vl_hat = get_f_pred(params, x_vl)

    print(sake.utils.bootstrap_mae(f_vl_hat, f_vl))
    # print(sake.utils.bootstrap_mae(f_vl_hat, f_vl))

if __name__ == "__main__":
    run("malonaldehyde")
