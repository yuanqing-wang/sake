import numpy as np
np.random.seed(2666)
import pickle

def run():
    data = pickle.load(open("data.npy", "rb"))
    i, x, y = data["i"], data["x"], data["y"]
    idxs = np.arange(len(i))
    np.random.shuffle(idxs)
    idxs_tr, idxs_vl, idxs_te = np.split(idxs, (int(0.85 * len(idxs)), int(0.90 * len(idxs))))
    i_tr, x_tr, y_tr = [i[idx] for idx in idxs_tr], [x[idx] for idx in idxs_tr], [y[idx] for idx in idxs_tr]
    i_vl, x_vl, y_vl = [i[idx] for idx in idxs_vl], [x[idx] for idx in idxs_vl], [y[idx] for idx in idxs_vl]
    i_te, x_te, y_te = [i[idx] for idx in idxs_te], [x[idx] for idx in idxs_te], [y[idx] for idx in idxs_te]

    
    ds_tr = [i_tr, x_tr, y_tr]
    ds_vl = [i_vl, x_vl, y_vl]
    ds_te = [i_te, x_te, y_te]

    pickle.dump(ds_tr, open("ds_tr.npy", "wb"), protocol=4)
    pickle.dump(ds_vl, open("ds_vl.npy", "wb"), protocol=4)
    pickle.dump(ds_te, open("ds_te.npy", "wb"), protocol=4)

if __name__ == "__main__":
    run()
