import numpy as np
import h5py

ELEMENT_ENERGY = {
    "H": -0.500607632585,
    "C": -37.8302333826,
    "N": -54.5680045287,
    "O": -75.0362229210,
}

ELEMENT = {
    "H": 1,
    "C": 2,
    "N": 3,
    "O": 4,
}

MAX_LENGTH = 26

def get_data():
    _i, _x, _y  = [], [], []
    datasets = [h5py.File("ANI-1_release/ani_gdb_s0%s.h5" % idx) for idx in range(1, 9)]
    for dataset in datasets:
        dataset_name = next(iter(dataset.keys()))
        for entry in dataset[dataset_name]:
            entry = dataset[dataset_name][entry]
            elements = entry['species']
            elements = [element.decode("UTF-8") for element in elements]
            offset = sum([ELEMENT_ENERGY[element] for element in elements])

            x = np.array(entry['coordinates'])
            y = np.array(entry['energies']) - offset
            i = entry['species']
            i = [_i.decode("UTF-8") for _i in i]
            i = np.array([ELEMENT[_i] for _i in i])
            i = np.repeat(np.expand_dims(i, 0), x.shape[0], 0)
            _i.append(i)
            _x.append(x)
            _y.append(y)

    data = {"i": _i, "x": _x, "y": _y}
    return data

if __name__ == "__main__":
    data = get_data()
    import pickle
    pickle.dump(data, open("data.npy", "wb"), protocol=4)


