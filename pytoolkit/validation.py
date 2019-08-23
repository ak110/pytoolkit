"""CVなど。"""
import numpy as np
import sklearn.model_selection

# import pytoolkit as tk


def split(dataset, nfold, split_seed=1, stratify=None):
    """nfold CV。"""
    if dataset.groups is not None:
        g = np.unique(dataset.groups)
        cv = sklearn.model_selection.KFold(
            n_splits=nfold, shuffle=True, random_state=split_seed
        )
        folds = []
        for train_indices, val_indices in cv.split(g, g):
            folds.append(
                (
                    np.where(np.in1d(dataset.groups, g[train_indices]))[0],
                    np.where(np.in1d(dataset.groups, g[val_indices]))[0],
                )
            )
    else:
        if stratify is None:
            stratify = (
                isinstance(dataset.labels, np.ndarray)
                and len(dataset.labels.shape) == 1
            )
        cv = (
            sklearn.model_selection.StratifiedKFold
            if stratify
            else sklearn.model_selection.KFold
        )
        cv = cv(nfold, shuffle=True, random_state=split_seed)
        folds = list(cv.split(dataset.data, dataset.labels))

    return folds
