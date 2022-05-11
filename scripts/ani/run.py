import torch
import jax
import jax.numpy as jnp
import numpy as onp
from torch.utils.data import DataLoader, Dataset
import h5py

DATA_NAMES = [
    "ani_gdb_s0%s.h5" idx % for idx in range(1, 9)
]

ELEMENT_ENERGY = {
    "H": -0.500607632585,
    "C": -37.8302333826,
    "N": -54.5680045287,
    "O": -75.0362229210,
}

class ANIDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.datasets = [h5py.File(data_name) for data_name in DATA_NAMES]

    @staticmethod
    def get_length(dataset):
        key = dataset.keys()[0]
        return len(dataset[key].keys())

    def _prepare(self):
        lengths = onp.array([self.get_length(dataset) for dataset in self.datasets])
        self.total_length = sum(lengths)
        self.length_bin = onp.cumsum(lengths)

    def __getitem__(self, idx):
        assert isinstance(idx, int), "Can only index by int."
        previous_idx = 0
        for count, length_bin_end in enumerate(self.length_bin):
            if length_bin_start > idx:
                dataset = self.datasets[count]
                sub_idx = idx - previous_idx
                break
            previous_idx = length_bin_end
        data = dataset[dataset.keys()[0]][dataset.keys()[0] + "-%s" % sub_idx]
        elements = data['species']
        elements = [element.decode("UTF-8") for element in elements]
        offset = sum([ELEMENT_ENERGY[element] for element in elements])
        return jnp.array(data['coordinates']), jnp.array(data['energies'] - offset)
