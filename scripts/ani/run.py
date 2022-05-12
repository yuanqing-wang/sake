import torch
import jax
import jax.numpy as jnp
import numpy as onp
from torch.utils.data import DataLoader, Dataset
import h5py

DATA_NAMES = [
    "ani_gdb_s0%s.h5" % idx for idx in range(1, 9)
]

ELEMENT_ENERGY = {
    "H": -0.500607632585,
    "C": -37.8302333826,
    "N": -54.5680045287,
    "O": -75.0362229210,
}

ELEMENT = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
}


class ANIDataset(object):
    def __init__(self):
        super().__init__()
        self.datasets = [h5py.File("ani_gdb_s0%s.h5" % idx) for idx in range(1, 9)]
        self.data_names = ["gdb11_s0%s" % idx for idx in range(1, 9)]
        self.keys = {data_name: list(dataset[data_name].keys()) for dataset, data_name in zip(self.datasets, self.data_names)}
        lengths = onp.array([len(value) for key, value in self.keys.items()])
        self.total_length = sum(lengths)
        self.length_bin = onp.cumsum(lengths)

    def __len__(self): return self.total_length

    def __getitem__(self, idx):
        assert isinstance(idx, int), "Can only index by int."
        previous_idx = 0
        for count, length_bin_end in enumerate(self.length_bin):
            if length_bin_end > idx:
                dataset = self.datasets[count]
                sub_idx = idx - previous_idx
                break
            previous_idx = length_bin_end
        data = self.datasets[count]
        data = data[self.data_names[count]][self.keys[self.data_names[count]][sub_idx]]

        elements = data['species']
        elements = [element.decode("UTF-8") for element in elements]
        offset = sum([ELEMENT_ENERGY[element] for element in elements])

        x = jnp.array(data['coordinates'])
        y = jnp.array(data['energies']) - offset
        y = jnp.expand_dims(y, -1)

        elements = jnp.array([ELEMENT[element] for element in elements])
        elements = jax.nn.one_hot(elements, 4)
        elements = jnp.broadcast_to(elements, (*x.shape[:-1], elements.shape[-1]))

        return elements, x, y


def run():
    data = ANIDataset()
    idxs = onp.arange(len(data))
    onp.random.seed(2666)
    onp.random.shuffle(idxs)
    idxs_tr, idxs_vl, idxs_te = onp.split(idxs, (int(0.8 * len(idxs)), int(0.05 * len(idxs))))

    import sake
    model = sake.models.DenseSAKEModel(
        hidden_features=64,
        out_features=1,
        depth=8,
    )

    def get_y_pred(params, i, x):
        y_pred, _, __ = model.apply(params, i, x)
        y_pred = y_pred.sum(axis=-2)
        return y_pred

    def get_loss(params, i, x, y):
        y_pred = get_y_pred(params,i, x)
        loss = jnp.abs(y_pred - y).mean()
        return loss

    i, x, y = data[idxs_tr[0].item()]
    params = model.init(jax.random.PRNGKey(2666), i, x)

    import optax 
    optimizer = optax.chain(
        optax.additive_weight_decay(1e-12),
        optax.clip(1.0),
        optax.adam(1e-5),
    )

    from flax.training.train_state import TrainState
    from flax.training.checkpoints import save_checkpoint

    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer,
    )

    @jax.jit
    def step(state, i, x, y):
        params = state.params
        loss, grads = jax.value_and_grad(get_loss)(params, i, x, y)
        state = state.apply_gradients(grads=grads)
        return loss, state

    from tqdm import tqdm
    for idx_batch in tqdm(range(50)):
        onp.random.seed(idx_batch)
        onp.random.shuffle(idxs_tr)
        for idx_data in idxs_tr:
            i, x, y = data[idx_data.item()]
            loss, state = step(state, i, x, y)
            print(loss)





    

if __name__ == "__main__":
    run()
