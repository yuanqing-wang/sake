import jax
import jax.numpy as jnp
import optax
import numpy as onp
import sake

def run(data):
    data = onp.load("%s_dft.npz" % data)
    onp.random.seed(2666)
    idxs = onp.random.permutation(len(data['R']))

    x = jnp.array(data['R'][idxs])
    e = jnp.array(data['E'][idxs])
    i = jnp.array(data['z'])
    f = jnp.array(data['F'][idxs])

    i = jnp.expand_dims(jax.nn.one_hot(i, i.max()), 0)
    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        depth=8,
        out_features=1,
    )

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
    coloring = partial(coloring, mean=x_tr.mean(), std=x_tr.std())
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-3,
        warmup_steps=1000 * n_batches,
        decay_steps=9000 * n_batches,
    )

    optimizer = optax.chain(
        optax.additive_weight_decay(1e-12),
        optax.adam(learning_rate=scheduler),
    )

    def get_e_pred(params, x):
        i_tr = jnp.repeat(i, x.shape[0], 0)
        e_pred, _, __ = model.apply(params, i_tr, x)
        e_pred = e_pred.sum(axis=1)
        e_pred = coloring(e_pred)
        return e_pred

    def get_e_pred_sum(params, x):
        return get_e_pred(params, x).sum()

    get_f_pred = jax.grad(get_e_pred_sum, argnums=(1,))

    def loss_fn(params, x, e, f):
        e_pred = get_e_pred(params, x)
        f_pred = get_f_pred(params, x)[0]
        e_loss = jnp.abs(e_pred - e).mean()
        f_loss = jnp.abs(f_pred - f).mean()
        return f_loss + e_loss * 0.001

    @jax.jit
    def step(params, opt_state, x, e, f):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, e, f)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def epoch(params, opt_state, x_tr, e_tr, f_tr):
        idxs = onp.random.permutation(len(x_tr))
        for idx_batch in range(n_batches):
            x = x_tr[idxs[idx_batch*batch_size:(idx_batch+1)*batch_size]]
            e = e_tr[idxs[idx_batch*batch_size:(idx_batch+1)*batch_size]]
            f = f_tr[idxs[idx_batch*batch_size:(idx_batch+1)*batch_size]]
            params, opt_state, loss = step(params, opt_state, x, e, f)
        return params, opt_state, loss

    i_tr = jnp.repeat(i, batch_size, 0)
    x0 = x_tr[:batch_size]
    params = model.init(jax.random.PRNGKey(2666), i_tr, x0)
    opt_state = optimizer.init(params)
    for idx_batch in range(10000):
        params, opt_state, loss_value = epoch(params, opt_state, x_tr, e_tr, f_tr)
        if idx_batch % 10 == 0:
            e_pred_vl = get_e_pred(params, x_vl)
            f_pred_vl = get_f_pred(params, f_vl)
            f1_e = jnp.abs(e_pred_vl - e_vl).mean()
            f1_f = jnp.abs(f_pred_vl - f_vl).mean()
            print(f1_e, f1_f)

if __name__ == "__main__":
    run("malonaldehyde")
