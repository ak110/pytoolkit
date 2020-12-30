import numpy as np
import sklearn.datasets

import pytoolkit as tk


def test_lgb(tmpdir):
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    folds = tk.validation.split(tk.data.Dataset(X, y), nfold=5)
    tk.lgb.cv(
        str(tmpdir),
        X,
        y,
        folds=folds,
        params={
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": 0.01,
            "nthread": 1,
            "verbosity": -1,
        },
    )
    boosters = tk.lgb.load(str(tmpdir), nfold=len(folds))
    pred = np.mean(tk.lgb.predict(boosters, X), axis=0).argmax(axis=-1)
    assert (pred == y).mean() >= 0.95
