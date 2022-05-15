import numpy as np
np.random.seed(2666)

def run():
    data = np.load("data.npy", allow_pickle=True)[()]
    lengths = list(data.keys())
    ds_tr, ds_vl, ds_te = {}, {}, {}
    for length in lengths:
        idxs = np.arange(data[length]['x'].shape[0])
        np.random.shuffle(idxs)
        idxs_tr, idxs_vl, idxs_te = np.split(idxs, (int(0.85 * len(idxs)), int(0.90 * len(idxs))))

        ds_tr[length], ds_vl[length], ds_te[length] = {}, {}, {}
        ds_tr[length]['i'], ds_tr[length]['x'], ds_tr[length]['y'] = data[length]['i'][idxs_tr], data[length]['x'][idxs_tr], data[length]['y'][idxs_tr]
        ds_vl[length]['i'], ds_vl[length]['x'], ds_vl[length]['y'] = data[length]['i'][idxs_vl], data[length]['x'][idxs_vl], data[length]['y'][idxs_vl]
        ds_te[length]['i'], ds_te[length]['x'], ds_te[length]['y'] = data[length]['i'][idxs_te], data[length]['x'][idxs_te], data[length]['y'][idxs_te]

    np.save("ds_tr", ds_tr)
    np.save("ds_vl", ds_vl)
    np.save("ds_te", ds_te)

if __name__ == "__main__":
    run()
