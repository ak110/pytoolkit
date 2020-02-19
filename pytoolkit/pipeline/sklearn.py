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
        models_dir: 保存先ディレクトリ
        weights_arg_name: tk.data.Dataset.weightsを使う場合の引数名
                          (pipelineなどで変わるので。例: "transformedtargetregressor__sample_weight")
        predict_method: "predict" or "predict_proba"

    """

    def __init__(
        self,
        estimator: sklearn.base.BaseEstimator,
        nfold: int,
        models_dir: tk.typing.PathLike,
        weights_arg_name: str = "sample_weight",
        predict_method: str = "predict",
        preprocessors: tk.pipeline.EstimatorListType = None,
        postprocessors: tk.pipeline.EstimatorListType = None,
    ):
        super().__init__(nfold, models_dir, preprocessors, postprocessors)
        self.estimator = estimator
        self.weights_arg_name = weights_arg_name
        self.predict_method = predict_method
        self.estimators_: typing.Optional[
            typing.List[sklearn.base.BaseEstimator]
        ] = None

    def _save(self, models_dir: pathlib.Path):
        tk.utils.dump(self.estimators_, models_dir / "estimators.pkl")

    def _load(self, models_dir: pathlib.Path):
        self.estimators_ = tk.utils.load(models_dir / "estimators.pkl")

    def _cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType) -> None:
        scores = []
        score_weights = []
        self.estimators_ = []
        for train_set, val_set in tk.utils.tqdm(dataset.iter(folds), desc="cv"):
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

        tk.log.get(__name__).info(
            f"cv score: {np.average(scores, weights=score_weights):,.3f}"
        )

    def _predict(self, dataset: tk.data.Dataset, fold: int) -> np.ndarray:
        assert self.estimators_ is not None
        if self.predict_method == "predict":
            return self.estimators_[fold].predict(dataset.data)
        elif self.predict_method == "predict_proba":
            return self.estimators_[fold].predict_proba(dataset.data)
        else:
            raise ValueError(f"predict_method={self.predict_method}")
