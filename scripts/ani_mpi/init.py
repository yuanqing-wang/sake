import jax
import jax.numpy as jnp
import numpy as onp
import flax
from functools import partial
import sake

BATCH_SIZE = 128
from run_tpu import Collater

def run():
    ds_tr = onp.load("ds_tr.npy", allow_pickle=True)[()]
    collater = Collater(ds_tr)

    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=6,
    )

    i, x, y = next(iter(collater))
    params = model.init(jax.random.PRNGKey(2666), i, x)

    import optax
    n_batches = len(collater)

    optimizer = optax.chain(
        optax.additive_weight_decay(1e-8),
        optax.clip(1.0),
        optax.zero_nans(),
        optax.adam(1e-5),
    )

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint

    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    save_checkpoint("_checkpoint", target=state, step=-1)

if __name__ == "__main__":
    run()
