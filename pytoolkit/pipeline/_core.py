"""パイプライン。"""
import pathlib

import sklearn.pipeline
import numpy as np

import pytoolkit as tk


class Model:
    """パイプラインのモデルのインターフェース。"""

    def cv(self, dataset, folds, models_dir):
        """CVして保存。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex
            models_dir (pathlib.Path): 保存先ディレクトリ (Noneなら保存しない)

        Returns:
            dict: metrics名と値

        """
        raise NotImplementedError()

    def load(self, models_dir):
        """読み込み。

        Args:
            models_dir (pathlib.Path): 保存先ディレクトリ

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
        pred_list = self.predict(dataset)
        assert len(pred_list) == len(folds)

        oofp_shape = (len(dataset),) + pred_list[0].shape[1:]
        oofp = np.empty(oofp_shape, dtype=pred_list[0].dtype)
        for pred, (_, val_indices) in zip(pred_list, folds):
            oofp[val_indices] = pred[val_indices]

        return oofp

    def predict(self, dataset):
        """予測結果をリストで返す。

        Args:
            dataset (tk.data.Dataset): 入力データ

        Returns:
            np.ndarray: len(self.folds)個の予測結果

        """
        raise NotImplementedError()


class Pipeline(Model):
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

    def cv(self, dataset, folds, models_dir):
        """CVして保存。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex
            models_dir (pathlib.Path): 保存先ディレクトリ (Noneなら保存しない)

        Returns:
            dict: metrics名と値

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
        scores = self.model.cv(dataset, folds, models_dir)

        if models_dir is not None:
            if self.preprocessors is not None:
                tk.utils.dump(self.preprocessors, models_dir / "preprocessors.pkl")
            if self.postprocessors is not None:
                tk.utils.dump(self.postprocessors, models_dir / "postprocessors.pkl")

        return scores

    def load(self, models_dir):
        """読み込み。

        Args:
            models_dir (PathLike): 保存先ディレクトリ

        Returns:
            Pipeline: self

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
        return self

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

        pred_list = self.model.predict(dataset)

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
