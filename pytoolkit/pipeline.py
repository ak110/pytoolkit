"""前処理＋モデル＋後処理のパイプライン。"""
import pathlib

import sklearn.pipeline
import numpy as np

import pytoolkit as tk


class Model:
    """パイプラインのモデルのインターフェース。"""

    def cv(self, dataset, folds):
        """CV。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex

        Returns:
            dict: metrics名と値

        """
        raise NotImplementedError()

    def predict_oof(self, dataset, folds):
        """out-of-foldなpredict結果を返す。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex

        Returns:
            np.ndarray: 予測結果

        """
        raise NotImplementedError()

    def predict(self, dataset):
        """予測結果をリストで返す。

        Args:
            dataset (tk.data.Dataset): 入力データ

        Returns:
            np.ndarray: len(self.folds)個の予測結果

        """
        raise NotImplementedError()

    def save(self, models_dir):
        """保存。

        Args:
            models_dir (pathlib.Path): 保存先ディレクトリ

        """
        raise NotImplementedError()

    def load(self, models_dir):
        """読み込み。

        Args:
            models_dir (pathlib.Path): 保存先ディレクトリ

        """
        raise NotImplementedError()


class Pipeline:
    """前処理＋モデル＋後処理のパイプライン。

    Args:
        model (Model): モデル
        preprocessors (list): 前処理 (sklearnのTransformerの配列)
        postprocessors (list): 後処理 (sklearnのTransformerの配列)

    """

    def __init__(self, model, preprocessors=None, postprocessors=None):
        self.model = model
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

    def cv(self, dataset, folds):
        """CV。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex

        Returns:
            dict: metrics名と値

        """
        dataset = dataset.copy()
        if self.preprocessors is not None:
            dataset.data = self.preprocessors.fit_transform(dataset.data, dataset.label)
        if self.postprocessors is not None:
            dataset.label = np.squeeze(
                self.postprocessors.fit_transform(
                    np.expand_dims(dataset.label, axis=-1)
                ),
                axis=-1,
            )
        self.model.cv(dataset, folds)

    def predict_oof(self, dataset, folds):
        """out-of-foldなpredict結果を返す。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex

        Returns:
            np.ndarray: 予測結果

        """
        dataset = dataset.copy()
        if self.preprocessors is not None:
            dataset.data = self.preprocessors.transform(dataset.data)
        oofp = self.model.predict_oof(dataset, folds)
        if self.postprocessors is not None:
            oofp = np.squeeze(
                self.postprocessors.inverse_transform(np.expand_dims(oofp, axis=-1)),
                axis=-1,
            )
        return oofp

    def predict(self, dataset):
        """予測結果をリストで返す。

        Args:
            dataset (tk.data.Dataset): 入力データ

        Returns:
            np.ndarray: len(self.folds)個の予測結果

        """
        dataset = dataset.copy()
        if self.preprocessors is not None:
            dataset.data = self.preprocessors.transform(dataset.data)
        predict_list = self.model.predict(dataset)
        if self.postprocessors is not None:
            for i in range(len(predict_list)):
                predict_list[i] = np.squeeze(
                    self.postprocessors.inverse_transform(
                        np.expand_dims(predict_list[i], axis=-1)
                    ),
                    axis=-1,
                )
        return predict_list

    def save(self, models_dir):
        """保存。

        Args:
            models_dir (PathLike): 保存先ディレクトリ

        """
        models_dir = pathlib.Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        if self.preprocessors is not None:
            tk.utils.dump(self.preprocessors, models_dir / "preprocessors.pkl")
        if self.postprocessors is not None:
            tk.utils.dump(self.postprocessors, models_dir / "postprocessors.pkl")
        self.model.save(models_dir)

    def load(self, models_dir):
        """読み込み。

        Args:
            models_dir (PathLike): 保存先ディレクトリ

        """
        models_dir = pathlib.Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessors = tk.utils.load(
            models_dir / "preprocessors.pkl", skip_not_exist=True
        )
        self.postprocessors = tk.utils.load(
            models_dir / "postprocessors.pkl", skip_not_exist=True
        )
        self.model.load(models_dir)
