import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
import tqdm

def run():
    ds_tr, ds_vl, ds_te = onp.load("train.npz"), onp.load("valid.npz"), onp.load("test.npz")
    i_tr, i_vl, i_te = ds_tr["charges"], ds_vl["charges"], ds_te["charges"]
    x_tr, x_vl, x_te = ds_tr["positions"], ds_vl["positions"], ds_te["positions"]
    
    m_tr, m_vl, m_te = (i_tr > 0).sum(-1) == 19, (i_vl > 0).sum(-1) == 19, (i_te > 0).sum(-1) == 19

    i_tr, i_vl, i_te = i_tr[m_tr, :19], i_vl[m_vl, :19], i_te[m_te, :19]
    x_tr, x_vl, x_te = x_tr[m_tr, :19, :], x_vl[m_vl, :19, :], x_te[m_te, :19, :]
    i_tr, i_vl, i_te = jax.nn.one_hot(i_tr, i_tr.max()+1), jax.nn.one_hot(i_vl, i_tr.max()+1), jax.nn.one_hot(i_te, i_tr.max()+1)

    BATCH_SIZE = 16
    N_BATCHES = len(i_tr) // BATCH_SIZE

    model = sake.flows.AugmentedFlowModel(depth=3, mp_depth=3)
    key = jax.random.PRNGKey(2666)
    h = jnp.zeros((BATCH_SIZE, 19, 10))
    
    print(x_tr.shape, i_tr.shape)
    params = model.init(key, h, x_tr[:BATCH_SIZE], x_tr[:BATCH_SIZE])
    prior = sake.flows.CenteredGaussian

    # @jax.jit
    def get_loss(params, key, i, x):
        v = prior.sample(key=key, shape=x.shape)
        v0 = v
        x, v, sum_log_det = model.apply(params, i, x, v, method=model.f_backward)
        loss = (-prior.log_prob(x) - prior.log_prob(v) + sum_log_det + prior.log_prob(v0)).mean()
        return loss

    from flax.training.checkpoints import restore_checkpoint
    params = restore_checkpoint("__checkpoint", target=None)["params"]

    def _get_loss(inputs):
        i = inputs[:, :-3]
        x = inputs[:, -3:]
        return get_loss(params, key, i, x)

    inputs = jnp.concatenate([i_vl, x_vl], -1)
    loss = jnp.mean(jax.lax.map(_get_loss, inputs))
    print(loss)

if __name__ == "__main__":
    import sys
    run()
