import sys
import os
import jax
import jax.numpy as jnp
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

    data_train = jnp.array(data_train.numpy())
    data_val = jnp.array(data_train.numpy())
    data_test = jnp.array(data_test.numpy())

    model = sake.models.DenseSAKEModel(width=64, depth=8)
    key = jax.random.PRNGKey(2666)
    params = model.init(key, data_train)

    prior = sake.flows.CenteredGaussian

    def loss(params):
        z, trace = sake.flows.ODEFlow.call(model, params)
        log_pz = prior.log_prob(z)
        log_px = (log_pz + trace).mean()
        loss = -log_px
        return loss

    loss(params)

    # optimizer = optax.adam(1e-4)
    # from flax.training.train_state import TrainState
    # from flax.training.checkpoints import save_checkpoint
    # state = TrainState.create(
    #     apply_fn=model.apply, params=params, tx=optimizer,
    # )

if __name__ == "__main__":
    run()
