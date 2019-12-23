"""scikit-learn"""
from __future__ import annotations

import pathlib
import typing

import numpy as np
import sklearn.base

import pytoolkit as tk

from .core import Model


class SKLearnModel(Model):
    """scikit-learnのモデル。

    Args:
        estimator: モデル
        nfold: cvの分割数
        weights_arg_name: tk.data.Dataset.weightsを使う場合の引数名
                          (pipelineなどで変わるので。例: "transformedtargetregressor__sample_weight")

    """

    def __init__(
        self,
        estimator: sklearn.base.BaseEstimator,
        nfold: int,
        weights_arg_name: str = "sample_weight",
        preprocessors: tk.pipeline.EstimatorListType = None,
        postprocessors: tk.pipeline.EstimatorListType = None,
    ):
        super().__init__(nfold, preprocessors, postprocessors)
        self.estimator = estimator
        self.weights_arg_name = weights_arg_name
        self.estimators_: typing.Optional[
            typing.List[sklearn.base.BaseEstimator]
        ] = None

    def _save(self, models_dir: pathlib.Path):
        tk.utils.dump(self.estimators_, models_dir / "estimators.pkl")

    def _load(self, models_dir: pathlib.Path):
        self.estimators_ = tk.utils.load(models_dir / "estimators.pkl")

    def _cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType) -> dict:
        scores = []
        score_weights = []
        self.estimators_ = []
        for train_indices, val_indices in tk.utils.tqdm(folds, desc="cv"):
            train_set = dataset.slice(train_indices)
            val_set = dataset.slice(val_indices)

            kwargs = {}
            if train_set.weights is not None:
                kwargs[self.weights_arg_name] = train_set.weights

            estimator = sklearn.base.clone(self.estimator)
            estimator.fit(train_set.data, train_set.labels, **kwargs)
            self.estimators_.append(estimator)

            kwargs = {}
            if val_set.weights is not None:
                kwargs[self.weights_arg_name] = val_set.weights

            scores.append(estimator.score(val_set.data, val_set.labels, **kwargs))
            score_weights.append(len(val_set))

        return {"score": np.average(scores, weights=score_weights)}

    def _predict(self, dataset: tk.data.Dataset, fold: int) -> np.ndarray:
        # TODO: predict_proba対応
        assert self.estimators_ is not None
        return self.estimators_[fold].predict(dataset.data)
