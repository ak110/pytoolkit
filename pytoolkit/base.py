"""アンサンブル。"""
import os
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

ModelType = typing.TypeVar("ModelType", bound="BaseModel")


class BaseModel(typing.Protocol):
    """弱学習器のインターフェース。"""

    def save(self, model_dir: str | os.PathLike[str]) -> None:
        """保存。

        Args:
            model_dir: 保存先ディレクトリ

        """

    @classmethod
    def load(cls: type[ModelType], model_dir: str | os.PathLike[str]) -> ModelType:
        """モデルの読み込み

        Args:
            model_dir: 保存先ディレクトリ

        Returns:
            モデル

        """

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

    def infers_to_labels(self, pred: npt.NDArray[np.float32]) -> npt.NDArray:
        """推論結果(infer, infer_oof)からクラス名などを返す。

        Args:
            pred: 推論結果

        Returns:
            クラス名など

        """
