import jax
import jax.numpy as jnp
import numpy as onp
import sake

def run(args):
    q_tr = jnp.load("charges_train_charged5_initvel1.npy")
    q_vl = jnp.load("charges_valid_charged5_initvel1.npy")
    q_te = jnp.load("charges_test_charged5_initvel1.npy")
    x_tr = jnp.load("loc_train_charged5_initvel1.npy")
    x_vl = jnp.load("loc_valid_charged5_initvel1.npy")
    x_te = jnp.load("loc_test_charged5_initvel1.npy")
    v_tr = jnp.load("vel_train_charged5_initvel1.npy")
    v_vl = jnp.load("vel_valid_charged5_initvel1.npy")
    v_te = jnp.load("vel_test_charged5_initvel1.npy")

    # x = x_tr.reshape((30, 100, *x_tr.shape[1:]))
    # q = q_tr.reshape((30, 100, *q_tr.shape[1:]))
    # v = v_tr.reshape((30, 100, *v_tr.shape[1:]))

    def preprocess(q, x, v):
        x = jnp.swapaxes(x, -2, -1)
        v = jnp.swapaxes(v, -2, -1)
        x0 = x[..., 30, :, :]
        x1 = x[..., 40, :, :]
        v = v[..., 30, :, :]
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        h = jnp.concatenate([q, v_norm], axis=-1)
        return h, x0, x1, v

    h_tr, x0_tr, x1_tr, v_tr = preprocess(q_tr, x_tr, v_tr)
    h_vl, x0_vl, x1_vl, v_vl = preprocess(q_vl, x_vl, v_vl)
    h_te, x0_te, x1_te, v_te = preprocess(q_te, x_te, v_te)

    model = sake.models.DenseSAKEModel(
        hidden_features=args.hidden_features,
        out_features=1,
        depth=args.depth,
        n_heads=args.n_heads,
        update=True,
    )

    params = model.init(jax.random.PRNGKey(2666), h_tr[0], x0_tr[0], v_tr[0])

    def loss_fn(params, h, x0, x1, v):
        _, x_hat, __ = model.apply(params, h, x0, v)
        return ((x_hat - x1) ** 2).mean()

    @jax.jit
    def step(state, h, x0, x1, v):
        params = state.params
        grads = jax.grad(loss_fn)(params, h, x0, x1, v)
        state = state.apply_gradients(grads=grads)
        return state

    # @jax.jit
    def epoch(state, key):
        idxs = jax.random.permutation(key, h_tr.shape[0])
        _h_tr, _x0_tr, _x1_tr, _v_tr = h_tr[idxs], x0_tr[idxs], x1_tr[idxs], v_tr[idxs]
        _h_tr = _h_tr.reshape((30, 100, *_h_tr.shape[1:]))
        _x0_tr = _x0_tr.reshape((30, 100, *_x0_tr.shape[1:]))
        _x1_tr = _x1_tr.reshape((30, 100, *_x1_tr.shape[1:]))
        _v_tr = _v_tr.reshape((30, 100, *_v_tr.shape[1:]))

        for h, x0, x1, v in zip(_h_tr, _x0_tr, _x1_tr, _v_tr):
            state = step(state, h, x0, x1, v)

        return state

    @jax.jit
    def eval(state):
        loss_vl = loss_fn(state.params, h_vl, x0_vl, x1_vl, v_vl)
        loss_te = loss_fn(state.params, h_te, x0_te, x1_te, v_te)
        return loss_vl, loss_te

    import optax

    optimizer = optax.chain(
        optax.additive_weight_decay(args.weight_decay),
        optax.adam(learning_rate=args.learning_rate),
    )

    from flax.training.train_state import TrainState
    state = TrainState.create(
        apply_fn = model.apply, params=params, tx=optimizer,
    )

    losses_vl = []
    losses_te = []

    key = jax.random.PRNGKey(2666)
    for _ in range(5000):
        key, subkey = jax.random.split(key)
        state = epoch(state, subkey)
        loss_vl, loss_te = eval(state)
        losses_vl.append(loss_vl.item())
        losses_te.append(loss_te.item())

    losses_vl = onp.array(losses_vl)
    losses_te = onp.array(losses_te)

    idx = losses_vl.argmin()
    print(args)
    print(losses_vl[idx], losses_te[idx])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()
    run(args)
