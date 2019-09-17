"""前処理＋モデル＋後処理のパイプライン。"""
from __future__ import annotations

import pathlib
import typing

import sklearn.pipeline
import numpy as np

import pytoolkit as tk


class Model:
    """パイプラインのモデルのインターフェース。

    Args:
        preprocessors: 前処理 (sklearnのTransformerの配列)
        postprocessors: 後処理 (sklearnのTransformerの配列)

    """

    def __init__(self, preprocessors: list = None, postprocessors: list = None):
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

    def cv(
        self,
        dataset: tk.data.Dataset,
        folds: tk.validation.FoldsType,
        models_dir: pathlib.Path,
    ) -> dict:
        """CVして保存。

        Args:
            dataset: 入力データ
            folds: CVのindex
            models_dir: 保存先ディレクトリ (Noneなら保存しない)

        Returns:
            metrics名と値

        """
        if models_dir is not None:
            models_dir = pathlib.Path(models_dir)
            models_dir.mkdir(parents=True, exist_ok=True)

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
        scores = self._cv(dataset, folds)

        if models_dir is not None:
            if self.preprocessors is not None:
                tk.utils.dump(self.preprocessors, models_dir / "preprocessors.pkl")
            if self.postprocessors is not None:
                tk.utils.dump(self.postprocessors, models_dir / "postprocessors.pkl")
            self._save(models_dir)

        return scores

    def load(self, models_dir) -> Model:
        """読み込み。

        Args:
            models_dir (PathLike): 保存先ディレクトリ

        Returns:
            self

        """
        models_dir = pathlib.Path(models_dir)
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
            予測結果

        """
        pred_list = self.predict(dataset)
        assert len(pred_list) == len(folds)

        oofp_shape = (len(dataset),) + pred_list[0].shape[1:]
        oofp = np.empty(oofp_shape, dtype=pred_list[0].dtype)
        for pred, (_, val_indices) in zip(pred_list, folds):
            oofp[val_indices] = pred[val_indices]

        return oofp

    def predict(self, dataset: tk.data.Dataset) -> typing.List[np.ndarray]:
        """予測結果をリストで返す。

        Args:
            dataset (tk.data.Dataset): 入力データ

        Returns:
            len(self.folds)個の予測結果

        """
        dataset = dataset.copy()
        if self.preprocessors is not None:
            dataset.data = self.preprocessors.transform(dataset.data)

        pred_list = self._predict(dataset)

        if self.postprocessors is not None:
            for i in range(len(pred_list)):
                if pred_list[i].ndim <= 1:
                    pred_list[i] = np.squeeze(
                        self.postprocessors.inverse_transform(
                            np.expand_dims(pred_list[i], axis=-1)
                        ),
                        axis=-1,
                    )
                else:
                    pred_list[i] = self.postprocessors.inverse_transform(pred_list[i])

        return pred_list

    def _save(self, models_dir):
        """保存。

        Args:
            models_dir (pathlib.Path): 保存先ディレクトリ

        """
        raise NotImplementedError()

    def _load(self, models_dir):
        """読み込み。

        Args:
            models_dir (pathlib.Path): 保存先ディレクトリ

        """
        raise NotImplementedError()

    def _cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType) -> dict:
        """CV。

        Args:
            dataset: 入力データ
            folds: CVのindex

        Returns:
            metrics名と値

        """
        raise NotImplementedError()

    def _predict(self, dataset: tk.data.Dataset) -> typing.List[np.ndarray]:
        """予測結果をリストで返す。

        Args:
            dataset (tk.data.Dataset): 入力データ

        Returns:
            len(self.folds)個の予測結果

        """
        raise NotImplementedError()
