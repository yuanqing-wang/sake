import sys
import os
import jax
import jax.numpy as jnp
import optax
sys.path.append(os.path.abspath("en_flows"))
from en_flows.dw4_experiment.dataset import get_data_dw4, remove_mean
import sake

def run():
    import numpy as np

    data_train, batch_iter_train = get_data_dw4(100, 'train', 100)
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

    model = sake.flows.AugmentedFlowModel()
    key = jax.random.PRNGKey(2666)
    params = model.init(key, data_train.sum(axis=-1, keepdims=True), data_train)

    prior = sake.flows.CenteredGaussian

    # @jax.jit
    def get_loss(params, key):
        x = data_train
        v = prior.sample(key=key, shape=x.shape)
        x, v, sum_log_det = model.apply(params, x, v, method=model.f_backward)
        loss = (-prior.log_prob(x) - prior.log_prob(v) + sum_log_det).mean()
        return loss

    def get_loss_vl(params, key):
        x = data_val
        v = prior.sample(key=key, shape=x.shape)
        x, v, sum_log_det = model.apply(params, x, v, method=model.f_backward)
        loss = (-prior.log_prob(x) - prior.log_prob(v) + sum_log_det).mean()
        return loss


    @jax.jit
    def step(state, key):
        params = state.params
        grads = jax.grad(get_loss)(params, key)
        state = state.apply_gradients(grads=grads)
        return state

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
        optax.adam(scheduler),
    )

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint
    state = TrainState.create(
         apply_fn=model.apply, params=params, tx=optimizer,
    )

    import tqdm
    for idx_batch in tqdm.tqdm(range(500)):
        key, subkey = jax.random.split(key)
        state = step(state, subkey)
        if idx_batch % 10 == 0:
            print(get_loss_vl(state.params, key), flush=True)


if __name__ == "__main__":
    run()
