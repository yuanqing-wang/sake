import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
import tqdm

def run(target):
 
    ds_tr, ds_vl, ds_te = onp.load("train.npz"), onp.load("valid.npz"), onp.load("test.npz")
    i_tr, i_vl, i_te = ds_tr["charges"], ds_vl["charges"], ds_te["charges"]
    x_tr, x_vl, x_te = ds_tr["positions"], ds_vl["positions"], ds_te["positions"]
    y_tr, y_vl, y_te = ds_tr[target], ds_vl[target], ds_te[target]
    
    if target + "_thermo" in ds_tr:
        y_tr = y_tr - ds_tr[target + "_thermo"]
        y_vl = y_vl - ds_vl[target + "_thermo"]
        y_te = y_te - ds_te[target + "_thermo"]

    y_tr, y_vl, y_te = onp.expand_dims(y_tr, -1), onp.expand_dims(y_vl, -1), onp.expand_dims(y_te, -1)
    m_tr, m_vl, m_te = (i_tr > 0), (i_vl > 0), (i_te > 0)

    def make_edge_mask(m):
        return jnp.expand_dims(m, -1) * jnp.expand_dims(m, -2)

    def sum_mask(m):
        return jnp.sign(m.sum(-1, keepdims=True))

    for _var in ["i", "x", "y", "m"]:
        for _split in ["tr", "vl", "te"]:
            locals()["%s_%s" % (_var, _split)] = jnp.array(locals()["%s_%s" % (_var, _split)])


    i_tr, i_vl, i_te = jax.nn.one_hot(i_tr, i_tr.max()+1), jax.nn.one_hot(i_vl, i_tr.max()+1), jax.nn.one_hot(i_te, i_tr.max()+1)
    m_tr, m_vl, m_te = make_edge_mask(m_tr), make_edge_mask(m_vl), make_edge_mask(m_te)

    BATCH_SIZE = 128
    N_BATCHES = len(i_tr) // BATCH_SIZE

    from sake.utils import coloring
    from functools import partial
    coloring = partial(coloring, mean=y_tr.mean(), std=y_tr.std())

    print(y_tr.mean(), y_tr.std())

    class Model(nn.Module):
        def setup(self):
            self.model = sake.models.DenseSAKEModel(
                hidden_features=64,
                out_features=64,
                depth=6,
                update=[False, False, False, True, True, True],
            )

            self.mlp = nn.Sequential(
                [
                    nn.Dense(64),
                    nn.silu,
                    nn.Dense(1),
                ],
            )

        def __call__(self, i, x, m):
            y, _, __ = self.model(i, x, mask=m)
            y = y * sum_mask(m)
            y = y.sum(-2)
            y = self.mlp(y)
            return y


    model = Model()

    def get_y_hat(params, i, x, m):
        y_hat = model.apply(params, i, x, m=m)
        y_hat = coloring(y_hat)
        return y_hat

    from flax.training.checkpoints import restore_checkpoint
    state = restore_checkpoint("_" + target, None)
    params = state['params']

    def _get_y_hat(inputs):
         x, i = jnp.split(inputs, [3], axis=-1)
         m = make_edge_mask(i.argmax(-1) > 0)
         return get_y_hat(params, i, x, m)
    
    inputs = jnp.concatenate([x_tr, i_tr], axis=-1)
    y_tr_hat = jax.lax.map(_get_y_hat, inputs)

    inputs = jnp.concatenate([x_vl, i_vl], axis=-1)
    y_vl_hat = jax.lax.map(_get_y_hat, inputs)

    inputs = jnp.concatenate([x_te, i_te], axis=-1)
    y_te_hat = jax.lax.map(_get_y_hat, inputs)

    print(y_tr_hat)
    
    print("training", sake.utils.bootstrap_mae(y_tr_hat, y_tr))
    print("validation", sake.utils.bootstrap_mae(y_vl_hat, y_vl))
    print("test", sake.utils.bootstrap_mae(y_te_hat, y_te))

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
