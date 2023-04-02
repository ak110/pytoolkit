"""アンサンブル。"""
import json
import os
import pathlib
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

import pytoolkit.base


class ModelMetadata(typing.TypedDict):
    """モデルのメタデータの型定義。"""

    num_models: int


class Model:
    """テーブルデータのモデル。"""

    def __init__(self, models: list[pytoolkit.base.BaseModel]):
        self.models = models
        self.metadata = ModelMetadata(num_models=len(self.models))

    def save(self, model_dir: str | os.PathLike[str]) -> None:
        """保存。

        Args:
            model_dir: 保存先ディレクトリ

        """
        model_dir = pathlib.Path(model_dir)
        for i, model in enumerate(self.models):
            model.save(model_dir / f"model{i}")
        (model_dir / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        model_dir: str | os.PathLike[str],
        model_type: type[pytoolkit.base.BaseModel],
    ) -> "Model":
        """モデルの読み込み

        Args:
            model_dir: 保存先ディレクトリ

        Returns:
            モデル

        """
        model_dir = pathlib.Path(model_dir)
        metadata: ModelMetadata = json.loads(
            (model_dir / "metadata.json").read_text(encoding="utf-8")
        )
        models = [
            model_type.load(model_dir / f"model{i}")
            for i in range(metadata["num_models"])
        ]
        return cls(models)

    def infer(
        self, data: pd.DataFrame | pl.DataFrame, verbose: bool = True
    ) -> npt.NDArray[np.float32]:
        """推論。

        Args:
            data: 入力データ
            verbose: 進捗表示の有無

        Returns:
            推論結果(分類ならshape=(num_samples,num_classes), 回帰ならshape=(num_samples,))

        """
        pred = np.mean(
            [model.infer(data, verbose) for model in self.models],
            axis=0,
            dtype=np.float32,
        )
        return pred

    def infer_oof(
        self,
        data: pd.DataFrame | pl.DataFrame,
        folds: typing.Sequence[tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]],
        verbose: bool = True,
    ) -> npt.NDArray[np.float32]:
        """out-of-fold推論。

        Args:
            data: 入力データ
            folds: 分割方法
            verbose: 進捗表示の有無

        Returns:
            推論結果(分類ならshape=(num_samples,num_classes), 回帰ならshape=(num_samples,))

        """
        pred = np.mean(
            [model.infer_oof(data, folds, verbose) for model in self.models],
            axis=0,
            dtype=np.float32,
        )
        return pred

    def infers_to_labels(self, pred: npt.NDArray[np.float32]) -> npt.NDArray:
        """推論結果(infer, infer_oof)からクラス名などを返す。

        Args:
            pred: 推論結果

        Returns:
            クラス名など

        """
        assert len(self.models) > 0
        return self.models[0].infers_to_labels(pred)
