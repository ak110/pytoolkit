"""CVなど。"""
from __future__ import annotations

import typing

import numpy as np
import sklearn.model_selection

import pytoolkit as tk

FoldsType = typing.Sequence[typing.Tuple[np.ndarray, np.ndarray]]


def split(
    dataset: tk.data.Dataset, nfold: int, split_seed: int = 1, stratify: bool = None
) -> FoldsType:
    """nfold CV。"""
    # nfold == 1は特別に5foldの最初の1個しか実行しないバージョンということにする
    # (もうちょっと分かりやすいインターフェースにしたいが利便性と両立する案が無いのでとりあえず…)
    if nfold == 1:
        return split(dataset, nfold=5, split_seed=split_seed, stratify=stratify)[:1]

    tk.log.get(__name__).info(
        f"split: len(dataset)={len(dataset)} nfold={nfold} split_seed={split_seed} stratify={stratify}"
    )

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

        if stratify:
            X = dataset.data
            y: typing.Any = dataset.labels
        else:
            X = list(range(len(dataset)))
            y = None

        cv = (
            sklearn.model_selection.StratifiedKFold
            if stratify
            else sklearn.model_selection.KFold
        )
        cv = cv(nfold, shuffle=True, random_state=split_seed)
        folds = list(cv.split(X, y))

    return folds


def pseudo_labeling(
    train_set: tk.data.Dataset,
    folds1: FoldsType,
    test_set: tk.data.Dataset,
    folds2: FoldsType,
    test_weights: float = 0.5,
):
    """pseudo labelなdataset, foldsを作って返す。

    Args:
        train_set: 訓練データ
        folds1: 訓練データのfolds
        test_set: テストデータ
        folds2: テストデータのfolds
        test_weights: 訓練データに対するテストデータの重み

    """
    dataset = tk.data.Dataset.concat(train_set, test_set)

    pl_weight = test_weights * len(train_set) / len(test_set)
    w_train = (
        np.ones(len(train_set)) if train_set.weights is None else train_set.weights
    )
    w_test = (
        np.ones(len(test_set)) if test_set.weights is None else test_set.weights
    ) * pl_weight
    dataset.weights = np.concatenate([w_train, w_test * pl_weight])

    # train_indicesをtrainとtestでconcatしたものにする。val_indicesはtrainのみ。
    folds = [
        (np.concatenate([f1_t, np.asarray(f2_t) + len(train_set)]), f1_v)
        for (f1_t, f1_v), (f2_t, _) in zip(folds1, folds2)
    ]
    return dataset, folds


def concat_folds(folds1: FoldsType, folds2: FoldsType, fold2_offset: int) -> FoldsType:
    """folds同士をくっつける。"""
    assert len(folds1) == len(folds2)

    return [
        (
            np.concatenate([f1_t, np.asarray(f2_t) + fold2_offset]),
            np.concatenate([f1_v, np.asarray(f2_v) + fold2_offset]),
        )
        for (f1_t, f1_v), (f2_t, f2_v) in zip(folds1, folds2)
    ]
