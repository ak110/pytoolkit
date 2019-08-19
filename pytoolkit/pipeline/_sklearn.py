import sklearn.base
import numpy as np

import pytoolkit as tk

from ._core import Model


class SKLearnModel(Model):
    """scikit-learnのモデル。

    Args:
        estimator (sklearn.base.BaseEstimator): モデル
        weights_arg_name (str): tk.data.Dataset.weightsを使う場合の引数名
                                (pipelineなどで変わるので。例: "transformedtargetregressor__sample_weight")

    """

    def __init__(self, estimator, weights_arg_name="sample_weight"):
        self.estimator = estimator
        self.weights_arg_name = weights_arg_name
        self.estimators_ = None

    def cv(self, dataset, folds, models_dir):
        """CVして保存。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex
            models_dir (pathlib.Path): 保存先ディレクトリ (Noneなら保存しない)

        Returns:
            dict: metrics名と値

        """
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

        if models_dir is not None:
            tk.utils.dump(self.estimators_, models_dir / "estimators.pkl")

        return {"score": np.average(scores, weights=score_weights)}

    def load(self, models_dir):
        """読み込み。

        Args:
            models_dir (pathlib.Path): 保存先ディレクトリ

        """
        self.estimators_ = tk.utils.load(models_dir / "estimators.pkl")

    def predict(self, dataset):
        """予測結果をリストで返す。

        Args:
            dataset (tk.data.Dataset): 入力データ

        Returns:
            np.ndarray: len(self.folds)個の予測結果

        """
        # TODO: predict_proba対応
        return np.array(
            [estimator.predict(dataset.data) for estimator in self.estimators_]
        )
