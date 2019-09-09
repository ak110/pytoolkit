"""CVなど。"""
import numpy as np
import sklearn.model_selection

import pytoolkit as tk


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


def pseudo_labeling(train_dataset, folds1, test_dataset, folds2, test_weights=0.5):
    """pseudo labelなdataset, foldsを作って返す。"""
    dataset = tk.data.Dataset.concat(train_dataset, test_dataset)

    pl_weight = test_weights * len(train_dataset) / len(test_dataset)
    w_train = (
        np.ones(len(train_dataset))
        if train_dataset.weights is None
        else train_dataset.weights
    )
    w_test = (
        np.ones(len(test_dataset))
        if test_dataset.weights is None
        else test_dataset.weights
    ) * pl_weight
    dataset.weights = np.concatenate([w_train, w_test * pl_weight])

    folds = concat_folds(folds1, folds2, len(train_dataset))
    return dataset, folds


def concat_folds(folds1, folds2, fold2_offset):
    """folds同士をくっつける。"""
    assert len(folds1) == len(folds2)

    return [
        tuple([np.concatenate([i1, i2 + fold2_offset]) for i1, i2 in zip(f1, f2)])
        for f1, f2 in zip(folds1, folds2)
    ]