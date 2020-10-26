import numpy as np
import sklearn.datasets

import pytoolkit as tk


def test_lgb():
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    folds = tk.validation.split(tk.data.Dataset(X, y), nfold=5)
    boosters, best_iteration = tk.lgb.cv(
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
    pred = np.mean(tk.lgb.predict(boosters, best_iteration, X), axis=0).argmax(axis=-1)
    assert (pred == y).mean() >= 0.95
