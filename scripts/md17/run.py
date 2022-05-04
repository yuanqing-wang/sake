import jax
import jax.numpy as jnp
import optax
import numpy as onp
import sake

def run(data):
    data = "malonaldehyde"
    data = onp.load("%s_dft.npz" % data)
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

    # from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=e_tr.mean(), std=e_tr.std())
    e_tr_mean = e_tr.mean().item()

    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=8,
    )

    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-3,
        warmup_steps=500 * n_batches,
        decay_steps=4500 * n_batches,
    )

    optimizer = optax.chain(
        optax.additive_weight_decay(1e-12),
        optax.clip(1.0),
        optax.adam(learning_rate=scheduler),
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

    def loss_fn(params, x, e, f):
        e_pred = get_e_pred(params, x)
        f_pred = get_f_pred(params, x)
        e_loss = jnp.abs(e_pred - e).mean()
        f_loss = jnp.abs(f_pred - f).mean()
        return f_loss + e_loss * 0.001

    @jax.jit
    def step(params, opt_state, x, e, f):
        grads = jax.grad(loss_fn)(params, x, e, f)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    @jax.jit
    def epoch(params, opt_state, x_tr, e_tr, f_tr, key):
        key, subkey = jax.random.split(key)
        idxs = jax.random.permutation(subkey, jnp.arange(x_tr.shape[0]))
        x_tr, e_tr, f_tr = x_tr[idxs], e_tr[idxs], f_tr[idxs]

        x_tr = x_tr.reshape(n_batches, batch_size, x_tr.shape[-2], x_tr.shape[-1])
        e_tr = e_tr.reshape(n_batches, batch_size, e_tr.shape[-1])
        f_tr = f_tr.reshape(n_batches, batch_size, f_tr.shape[-2], f_tr.shape[-1])

        def loop_body(idx_batch, state):
            params, opt_state = state
            x = x_tr[idx_batch]
            e = e_tr[idx_batch]
            f = f_tr[idx_batch]
            params, opt_state = step(params, opt_state, x, e, f)
            state = params, opt_state
            return state

        params, opt_state = jax.lax.fori_loop(0, n_batches, loop_body, (params, opt_state))

        return params, opt_state, key

    from functools import partial

    key = jax.random.PRNGKey(2666)
    i_tr = jnp.repeat(i, batch_size, 0)
    x0 = x_tr[:batch_size]
    params = model.init(jax.random.PRNGKey(2666), i_tr, x0)
    opt_state = optimizer.init(params)
    for idx_batch in range(500):
        import time
        time0 = time.time()
        params, opt_state, key = many_epochs(params, opt_state, x_tr, e_tr, f_tr, key)
        time1 = time.time()
        if idx_batch % 10 == 0:
            e_pred_vl = get_e_pred(params, x_vl[:100])
            f_pred_vl = get_f_pred(params, x_vl[:100])
            f1_e = jnp.abs(e_pred_vl - e_vl[:100]).mean()
            f1_f = jnp.abs(f_pred_vl - f_vl[:100]).mean()
            print("------------")
            print(idx_batch, time1 - time0)
            print(f1_e, f1_f)


if __name__ == "__main__":
    run("malonaldehyde")
