import lmdb
import pickle
import numpy as np

def run():
    path = "is2res_train_val_test_lmdbs/data/is2re/10k/train/data.lmdb"
    env = lmdb.open(
        path,
        subdir=False,
        readonly=True,
        lock=False,
    )
    length = env.stat()["entries"]
    keys = [f"{j}".encode("ascii") for j in range(env.stat()["entries"])]
    txn = env.begin()
    data = []

    for key in keys:
        _data = pickle.loads(txn.get(key)).__dict__
        _data = [
            _data["pos"].numpy(),
            _data["y_relaxed"],
            _data["atomic_numbers"].numpy(),
        ]

        data.append(_data)

    np.save("is2re10k.npy", data)

if __name__ == "__main__":
    run()
