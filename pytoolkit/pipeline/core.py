"""前処理＋モデル＋後処理のパイプライン。"""
from __future__ import annotations

import pathlib
import typing

import numpy as np
import sklearn.base
import sklearn.pipeline

import pytoolkit as tk

EstimatorListType = typing.Sequence[sklearn.base.BaseEstimator]


class Model:
    """パイプラインのモデルのインターフェース。

    Args:
        nfold: CVの分割数
        models_dir: 保存先ディレクトリ
        preprocessors: 前処理 (sklearnのTransformerの配列)
        postprocessors: 後処理 (sklearnのTransformerの配列)
        save_on_cv: cv時にsaveもするならTrue。

    """

    def __init__(
        self,
        nfold: int,
        models_dir: tk.typing.PathLike,
        preprocessors: EstimatorListType = None,
        postprocessors: EstimatorListType = None,
        save_on_cv: bool = True,
    ):
        self.nfold = nfold
        self.models_dir = pathlib.Path(models_dir)
        self.preprocessors = (
            sklearn.pipeline.make_pipeline(*preprocessors)
            if preprocessors is not None
            else None
        )
        self.postprocessors = (
            sklearn.pipeline.make_pipeline(*postprocessors)
            if postprocessors is not None
            else None
        )
        self.save_on_cv = save_on_cv

    def cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType) -> Model:
        """CVして保存。

        Args:
            dataset: 入力データ
            folds: CVのindex

        Returns:
            self

        """
        dataset = dataset.copy()
        if self.preprocessors is not None:
            dataset.data = self.preprocessors.fit_transform(
                dataset.data, dataset.labels
            )
        if self.postprocessors is not None:
            dataset.labels = np.squeeze(
                self.postprocessors.fit_transform(
                    np.expand_dims(dataset.labels, axis=-1)
                ),
                axis=-1,
            )

        self._cv(dataset, folds)
        if self.save_on_cv:
            self.save()

        return self

    def save(self, models_dir: tk.typing.PathLike = None) -> Model:
        """保存。

        Args:
            models_dir: 保存先ディレクトリ (Noneならself.models_dir)

        Returns:
            self

        """
        models_dir = pathlib.Path(models_dir or self.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        if self.preprocessors is not None:
            tk.utils.dump(self.preprocessors, models_dir / "preprocessors.pkl")
        if self.postprocessors is not None:
            tk.utils.dump(self.postprocessors, models_dir / "postprocessors.pkl")
        self._save(models_dir)
        return self

    def load(self, models_dir: tk.typing.PathLike = None) -> Model:
        """読み込み。

        Args:
            models_dir: 保存先ディレクトリ (Noneならself.models_dir)

        Returns:
            self

        """
        models_dir = pathlib.Path(models_dir or self.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessors = tk.utils.load(
            models_dir / "preprocessors.pkl", skip_not_exist=True
        )
        self.postprocessors = tk.utils.load(
            models_dir / "postprocessors.pkl", skip_not_exist=True
        )
        self._load(models_dir)
        return self

    def predict_oof(
        self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType
    ) -> np.ndarray:
        """out-of-foldなpredict結果を返す。

        Args:
            dataset: 入力データ
            folds: CVのindex

        Returns:
            推論結果

        """
        pred_list = [
            self.predict(dataset.slice(val_indices), fold)
            for fold, (_, val_indices) in enumerate(folds)
        ]
        assert len(pred_list) == len(folds)

        if isinstance(pred_list[0], list):  # multiple output
            oofp = [
                self._get_oofp(dataset, folds, [p[i] for p in pred_list])
                for i in range(len(pred_list[0]))
            ]
        else:
            oofp = self._get_oofp(dataset, folds, pred_list)
        return oofp

    def _get_oofp(self, dataset, folds, pred_list):
        oofp_shape = (len(dataset),) + pred_list[0].shape[1:]
        oofp = np.empty(oofp_shape, dtype=pred_list[0].dtype)
        for pred, (_, val_indices) in zip(pred_list, folds):
            oofp[val_indices] = pred
        return oofp

    def predict_all(self, dataset: tk.data.Dataset) -> list[np.ndarray]:
        """全fold分の推論結果をリストで返す。"""
        return [self.predict(dataset, fold) for fold in range(self.nfold)]

    def predict(self, dataset: tk.data.Dataset, fold: int) -> np.ndarray:
        """推論結果を返す。

        Args:
            dataset: 入力データ
            fold: 使用するモデル

        Returns:
            推論結果

        """
        dataset = dataset.copy()
        if self.preprocessors is not None:
            dataset.data = self.preprocessors.transform(dataset.data)

        pred = self._predict(dataset, fold)

        if self.postprocessors is not None:
            if isinstance(pred, np.ndarray) and pred.ndim <= 1:
                pred = np.squeeze(
                    self.postprocessors.inverse_transform(
                        np.expand_dims(pred, axis=-1)
                    ),
                    axis=-1,
                )
            else:
                pred = self.postprocessors.inverse_transform(pred)

        return pred

    def _save(self, models_dir: pathlib.Path):
        """保存。

        Args:
            models_dir: 保存先ディレクトリ

        """
        raise NotImplementedError()

    def _load(self, models_dir: pathlib.Path):
        """読み込み。

        Args:
            models_dir: 保存先ディレクトリ

        """
        raise NotImplementedError()

    def _cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType) -> None:
        """CV。

        Args:
            dataset: 入力データ
            folds: CVのindex

        """
        raise NotImplementedError()

    def _predict(self, dataset: tk.data.Dataset, fold: int) -> np.ndarray:
        """推論結果を返す。

        Args:
            dataset: 入力データ
            fold: 使用するモデル

        Returns:
            推論結果

        """
        raise NotImplementedError()
