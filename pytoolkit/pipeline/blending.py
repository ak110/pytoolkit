"""scikit-learn"""
from __future__ import annotations

import pathlib
import typing


import numpy as np

import pytoolkit as tk

from .core import Model


class BlendingModel(Model):
    """平均を取るだけのアンサンブル。

    TODO: 作りかけ。将来的に重み付き平均にして重みの推定とかも入れたい。

    Args:
        num_models: モデル数
        models_dir: 保存先ディレクトリ
        score_fn: 指定した場合は重みを探索する。指定しなければ全部同じ重み。
        direction: "minimize" or "maximize"

    """

    def __init__(
        self,
        num_models: int,
        models_dir: tk.typing.PathLike,
        score_fn: typing.Callable[[tk.data.LabelsType, np.ndarray], float] = None,
        direction: str = None,
        n_trials: int = 100,
        preprocessors: tk.pipeline.EstimatorListType = None,
        postprocessors: tk.pipeline.EstimatorListType = None,
    ):
        super().__init__(1, models_dir, preprocessors, postprocessors)
        if score_fn is not None:
            assert direction is not None, '"direction" is required'
        self.num_models = num_models
        self.score_fn = score_fn
        self.direction = direction
        self.n_trials = n_trials
        self.weights_: typing.Optional[typing.List[float]] = None

    def _save(self, models_dir: pathlib.Path):
        tk.utils.dump(self.weights_, models_dir / "weights.pkl")

    def _load(self, models_dir: pathlib.Path):
        self.weights_ = tk.utils.load(models_dir / "weights.pkl")

    def _cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType) -> None:
        del folds
        if self.score_fn is None:
            pass
        else:
            # optunaでブラックボックス探索する。(metrics次第ではもっと効率のいい方法がありそうだが手抜き)
            data = self._get_data(dataset)

            def params_fn(trial):
                w = np.array(
                    [
                        trial.suggest_uniform(f"w{i}", 1, 100)
                        for i in range(self.num_models)
                    ]
                )
                w *= self.num_models / w.sum()
                return {f"w{i}": w[i] for i in range(self.num_models)}

            def score_fn(params):
                weights = np.array([params[f"w{i}"] for i in range(self.num_models)])
                y_pred = np.average(data, axis=1, weights=weights)
                assert self.score_fn is not None
                return self.score_fn(dataset.labels, y_pred)

            study = tk.hpo.optimize(
                params_fn, score_fn, direction=self.direction, n_trials=100, n_jobs=-1,
            )
            best_params = tk.hpo.get_best_params(study, params_fn)
            self.weights_ = np.array(
                [best_params[f"w{i}"] for i in range(self.num_models)]
            )

    def _predict(self, dataset: tk.data.Dataset, fold: int) -> np.ndarray:
        assert isinstance(dataset.data, np.ndarray)
        assert dataset.data.ndim == 2
        assert (
            dataset.data.shape[1] % self.num_models == 0
        ), f"shape error: {dataset.data.shape}"
        data = self._get_data(dataset)
        return np.average(data, axis=1, weights=self.weights_)

    def _get_data(self, dataset):
        input_shape = (
            (len(dataset)),
            self.num_models,
            dataset.data.shape[1] // self.num_models,
        )
        data = dataset.data.reshape(input_shape)
        return data
