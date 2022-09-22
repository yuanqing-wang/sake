import sys
import os
import jax
import jax.numpy as jnp
import optax
sys.path.append(os.path.abspath("en_flows"))
from en_flows.dw4_experiment.dataset import get_data_dw4, remove_mean
import sake

def run(n_data):
    import numpy as np
    batch_size = 100

    data_train, batch_iter_train = get_data_dw4(n_data, 'train', 100)
    data_val, batch_iter_val = get_data_dw4(100, 'val', 100)
    data_test, batch_iter_test = get_data_dw4(100, 'test', 100)

    data_train = data_train.reshape(-1, 4, 2)
    data_val = data_val.reshape(-1, 4, 2)
    data_test = data_test.reshape(-1, 4, 2)

    data_train = data_train - data_train.mean(dim=-2, keepdim=True)
    data_val = data_val - data_val.mean(dim=-2, keepdim=True)
    data_test = data_test - data_test.mean(dim=-2, keepdim=True)

    data_train = jnp.array(data_train)
    data_val = jnp.array(data_train)
    data_test = jnp.array(data_test)

    n_batches = int(len(data_train) / batch_size)

    model = sake.flows.AugmentedFlowModel(depth=4, mp_depth=4)
    key = jax.random.PRNGKey(2666)
    h = jnp.zeros((batch_size, 4, 2))
    params = model.init(key, h, data_train[:100], data_train[:100])
    prior = sake.flows.CenteredGaussian

    # @jax.jit
    def get_loss(params, key, x):
        v = prior.sample(key=key, shape=x.shape)
        x, v, sum_log_det = model.apply(params, h, x, v, method=model.f_backward)
        loss = (-prior.log_prob(x) - prior.log_prob(v) + sum_log_det).mean()
        return loss

    @jax.jit
    def get_loss_vl(params, key):
        x = data_val
        v = prior.sample(key=key, shape=x.shape)
        v0 = v
        x, v, sum_log_det = model.apply(params, h, x, v, method=model.f_backward)
        loss = (-prior.log_prob(x) - prior.log_prob(v) + sum_log_det + prior.log_prob(v0)).mean()
        return loss

    # @jax.jit
    def step(state, key, x):
        key, subkey = jax.random.split(key)
        params = state.params
        grads = jax.grad(get_loss)(params, subkey, x)
        state = state.apply_gradients(grads=grads)
        return state, key

    def epoch(state, key):
        idxs = jax.random.permutation(key, len(data_train))
        _x = data_train[idxs].reshape(n_batches, batch_size, data_train.shape[-2], data_train.shape[-1])
        
        def fn(idx, _state):
            state, key = _state
            state, key = step(state, key, _x[idx])
            _state = state, key
            return _state

        state, key = jax.lax.fori_loop(0, n_batches, fn, (state, key))
        return state, key

    from functools import partial
    @partial(jax.jit, static_argnums=(2,))
    def many_epochs(state, key, n=100):
        def fn(idx, _state):
            state, key = _state
            state, key = epoch(state, key)
            _state = (state, key)
            return _state

        state, key = jax.lax.fori_loop(0, n, fn, (state, key))
        return state, key

    import optax
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-3,
        warmup_steps=50,
        decay_steps=450,
    )
    optimizer = optax.chain(
        optax.additive_weight_decay(1e-5),
        optax.clip(1.0),
        optax.adam(1e-5),
    )

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
         apply_fn=model.apply, params=params, tx=optimizer,
    )

    import tqdm
    for idx_batch in tqdm.tqdm(range(5000)):
        state, key = many_epochs(state, key)
        save_checkpoint("_" + str(n_data), target=state, step=idx_batch)


if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]))
