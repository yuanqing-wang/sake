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
    h = jnp.zeros((n_data, 4, 2))
    prior = sake.flows.CenteredGaussian

    @jax.jit
    def get_loss_vl(params, key):
        x = data_val
        v = prior.sample(key=key, shape=x.shape)
        v0 = v
        x, v, sum_log_det = model.apply(params, h, x, v, method=model.f_backward)
        loss = (-prior.log_prob(x) - prior.log_prob(v) + sum_log_det + prior.log_prob(v0)).mean()
        return loss

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import restore_checkpoint
    params = restore_checkpoint("_" + str(n_data), target=None)["params"]
    loss = get_loss_vl(params, jax.random.PRNGKey(2666))
    print(loss)

if __name__ == "__main__":
    import sys
    run(int(sys.argv[1]))
