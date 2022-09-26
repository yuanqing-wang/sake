import lmdb
import pickle
import numpy as np

def run():
    data = {}
    path = "../oc20/is2res_train_val_test_lmdbs/data/is2re/all/train/data.lmdb"
    env = lmdb.open(
        path,
        subdir=False,
        readonly=True,
        lock=False,
    )
    length = env.stat()["entries"]
    keys = [f"{j}".encode("ascii") for j in range(env.stat()["entries"])]
    txn = env.begin()

    for key in keys:
        _data = pickle.loads(txn.get(key)).__dict__
        x = _data["pos"].numpy()
        y = _data["y_relaxed"]
        i = _data["atomic_numbers"].numpy()
        i = np.repeat(np.expand_dims(i, 0), x.shape[0], 0)
        length = x.shape[0]
        if length in data:
            data[length]['i'].append(i)
            data[length]['x'].append(x)
            data[length]['y'].append(y)
        else:
            data[length] = {'i': [i], 'x': [x], 'y': [y]}

    for length in data:
        data[length]['i'] = np.stack(data[length]['i'])
        data[length]['x'] = np.stack(data[length]['x'])
        data[length]['y'] = np.stack(data[length]['y'])

    np.save("is2re_all.npy", data)

if __name__ == "__main__":
    run()
