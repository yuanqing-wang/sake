import numpy as np
import h5py

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


def get_data():
    data = []
    datasets = [h5py.File("ANI-1_release/ani_gdb_s0%s.h5" % idx) for idx in range(1, 9)]
    for dataset in datasets:
        dataset_name = next(iter(dataset.keys()))
        for entry in dataset[dataset_name]:
            entry = dataset[dataset_name][entry]
            smiles = "".join(entry["smiles"].asstr()[...])
            data.append(smiles)


    import json
    json.dump(data, open("smiles.pkl", "w"))

if __name__ == "__main__":
    data = get_data()
    np.save("data.npy", data)


