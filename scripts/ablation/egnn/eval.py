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
    i = jax.nn.one_hot(i, i.max())

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

    model = sake.models.EquivariantGraphNeuralNetwork(
        hidden_features=64,
        out_features=1,
        depth=6,
    )

    @jax.jit
    def get_e_pred(params, x):
        i_tr = jnp.broadcast_to(i, (*x.shape[:-1], i.shape[-1]))
        e_pred, _, __ = model.apply(params, i_tr, x)
        e_pred = e_pred.sum(axis=-2)
        e_pred = coloring(e_pred)
        return e_pred

    def get_e_pred_sum(params, x):
        e_pred = get_e_pred(params, x)
        return -e_pred.sum()

    get_f_pred = jax.jit(lambda params, x: jax.grad(get_e_pred_sum, argnums=(1,))(params, x)[0])

    from flax.training.checkpoints import restore_checkpoint
    state = restore_checkpoint("_" + data_name, None)
    params = state['params']

    _get_e_pred = lambda x: get_e_pred(params=params, x=x)
    _get_f_pred = lambda x: get_f_pred(params=params, x=x)

    def _get_e_pred(x):
        e_pred = get_e_pred(params, x)
        return e_pred

    e_vl_hat = jax.lax.map(_get_e_pred, x_vl)
    f_vl_hat = jax.lax.map(_get_f_pred, x_vl)
    e_te_hat = jax.lax.map(_get_e_pred, x_te)
    f_te_hat = jax.lax.map(_get_f_pred, x_te)

    # print("validation", sake.utils.bootstrap_mae(f_vl_hat, f_vl), sake.utils.bootstrap_mae(e_vl_hat, e_vl))
    # print("test", sake.utils.bootstrap_mae(f_te_hat, f_te), sake.utils.bootstrap_mae(e_te_hat, e_te))
    #
    origin, low, high = sake.utils.bootstrap_mae(f_te_hat, f_te)
    origin, low, high = 43.364 * origin, 43.364 * low, 43.364 * high
    print("force, $%.2f_{%.2f}^{%.2f}$" % (origin, low, high))


    origin, low, high = sake.utils.bootstrap_mae(e_te_hat, e_te)
    origin, low, high = 43.364 * origin, 43.364 * low, 43.364 * high
    print("energy, $%.2f_{%.2f}^{%.2f}$" % (origin, low, high))


if __name__ == "__main__":
    import sys
    run(sys.argv[1])
