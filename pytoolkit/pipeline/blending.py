"""scikit-learn"""
from __future__ import annotations

import pathlib

import numpy as np

import pytoolkit as tk

from .core import Model


class BlendingModel(Model):
    """平均を取るだけのアンサンブル。

    TODO: 作りかけ。将来的に重み付き平均にして重みの推定とかも入れたい。

    Args:
        num_models: モデル数
        nfold: cvの分割数
        models_dir: 保存先ディレクトリ

    """

    def __init__(
        self,
        num_models: int,
        nfold: int,
        models_dir: tk.typing.PathLike,
        preprocessors: tk.pipeline.EstimatorListType = None,
        postprocessors: tk.pipeline.EstimatorListType = None,
    ):
        super().__init__(nfold, models_dir, preprocessors, postprocessors)
        self.num_models = num_models

    def _save(self, models_dir: pathlib.Path):
        # tk.utils.dump(self.estimators_, models_dir / "estimators.pkl")
        pass

    def _load(self, models_dir: pathlib.Path):
        # self.estimators_ = tk.utils.load(models_dir / "estimators.pkl")
        pass

    def _cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType) -> None:
        # scores = []
        # score_weights = []
        # self.estimators_ = []
        # for train_set, val_set in tk.utils.tqdm(
        #     dataset.iter(folds), total=len(folds), desc="cv"
        # ):
        #     kwargs = {}
        #     if train_set.weights is not None:
        #         kwargs[self.weights_arg_name] = train_set.weights

        #     estimator = sklearn.base.clone(self.estimator)
        #     estimator.fit(train_set.data, train_set.labels, **kwargs)
        #     self.estimators_.append(estimator)

        #     kwargs = {}
        #     if val_set.weights is not None:
        #         kwargs[self.weights_arg_name] = val_set.weights

        #     scores.append(estimator.score(val_set.data, val_set.labels, **kwargs))
        #     score_weights.append(len(val_set))

        # tk.log.get(__name__).info(
        #     f"cv score: {np.average(scores, weights=score_weights):,.3f}"
        # )
        pass

    def _predict(self, dataset: tk.data.Dataset, fold: int) -> np.ndarray:
        assert isinstance(dataset.data, np.ndarray)
        assert dataset.data.ndim == 2
        assert (
            dataset.data.shape[1] % self.num_models == 0
        ), f"shape error: {dataset.data.shape}"
        input_shape = (
            (len(dataset)),
            self.num_models,
            dataset.data.shape[1] // self.num_models,
        )
        data = dataset.data.reshape(input_shape)
        return data.mean(axis=1)
