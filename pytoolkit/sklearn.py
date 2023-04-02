"""LightGBM関連。

importしただけでlightgbm.register_loggerを呼び出すため注意。

Examples:
    ::

        import pytoolkit.sklearn
        import pytoolkit.tablurs

        train_data_path = "path/to/train.csv"
        test_data_path = "path/to/test.csv"
        input_data_path = "path/to/input.csv"

        # ログの初期化
        pytoolkit.logs.init()

        # データの読み込み
        train_data, train_labels = pytoolkit.tablurs.load_labeled_data(
            train_data_path, "label_col_name"
        )
        test_data, test_labels = pytoolkit.tablurs.load_labeled_data(
            test_data_path, "label_col_name"
        )

        # 学習
        model = pytoolkit.sklearn.train(train_data, train_labels, groups=None)

        # 保存・読み込み
        model.save(model_dir)
        model = pytoolkit.sklearn.load(model_dir)

        # 評価
        evals = model.evaluate(test_data, test_labels)
        assert 0.0 <= evals["acc"] <= 1.0
        assert 0.0 <= evals["auc"] <= 1.0

        # 推論
        input_data = pytoolkit.tablurs.load_unlabeled_data(input_data_path)
        results = model.infer(input_data)
        assert isinstance(results, np.ndarray)

"""
import json
import logging
import os
import pathlib
import typing

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import sklearn.base
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import tqdm

import pytoolkit.base

logger = logging.getLogger(__name__)


class ModelMetadata(typing.TypedDict):
    """モデルのメタデータの型定義。"""

    task: typing.Literal["binary", "multiclass", "regression"]
    columns: list[str]
    class_names: list[typing.Any] | None
    nfold: int


class Model(pytoolkit.base.BaseModel):
    """テーブルデータのモデル。"""

    def __init__(
        self, estimators: list[sklearn.base.BaseEstimator], metadata: ModelMetadata
    ):
        self.estimators = estimators
        self.metadata = metadata

    def save(self, model_dir: str | os.PathLike[str]) -> None:
        """保存。

        Args:
            model_dir: 保存先ディレクトリ

        """
        model_dir = pathlib.Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimators, model_dir / "estimators.joblib")
        (model_dir / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, model_dir: str | os.PathLike[str]) -> "Model":
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
        estimators: list[sklearn.base.BaseEstimator] = joblib.load(
            model_dir / "estimators.joblib"
        )
        return cls(estimators, metadata)

    def evaluate(
        self, data: pd.DataFrame | pl.DataFrame, labels: npt.ArrayLike
    ) -> dict[str, float]:
        """推論。

        Args:
            data: 入力データ
            labels: ラベル

        Returns:
            スコア

        """
        pred = self.infer(data)
        if self.metadata["task"] == "binary":
            assert self.metadata["class_names"] is not None
            labels = _class_to_index(labels, self.metadata["class_names"])
            return {
                "acc": float(sklearn.metrics.accuracy_score(labels, np.round(pred))),
                "auc": float(sklearn.metrics.roc_auc_score(labels, pred)),
            }
        elif self.metadata["task"] == "multiclass":
            assert self.metadata["class_names"] is not None
            labels = _class_to_index(labels, self.metadata["class_names"])
            return {
                "acc": float(
                    sklearn.metrics.accuracy_score(labels, pred.argmax(axis=-1))
                ),
                "auc": float(
                    sklearn.metrics.roc_auc_score(labels, pred, multi_class="ovo")
                ),
            }
        else:
            assert self.metadata["task"] == "regression"
            return {
                "mae": float(sklearn.metrics.mean_absolute_error(labels, pred)),
                "rmse": float(
                    sklearn.metrics.mean_squared_error(labels, pred, squared=False)
                ),
                "r2": float(sklearn.metrics.r2_score(labels, pred)),
            }

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
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        data = data[self.metadata["columns"]].to_numpy()

        pred = np.mean(
            [
                estimator.predict(data)
                if self.metadata["task"] == "regression"
                else estimator.predict_proba(data)
                for estimator in tqdm.tqdm(
                    self.estimators,
                    ascii=True,
                    ncols=100,
                    desc="predict",
                    disable=not verbose,
                )
            ],
            axis=0,
            dtype=np.float32,
        )
        if self.metadata["task"] == "binary":
            pred = np.stack([1 - pred, pred], axis=-1)
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
        assert len(folds) == len(self.estimators)
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        data = data[self.metadata["columns"]].to_numpy()

        oofp: npt.NDArray[np.float32] | None = None
        for estimator, (_, val_indices) in tqdm.tqdm(
            list(zip(self.estimators, folds, strict=True)),
            ascii=True,
            ncols=100,
            desc="infer_oof",
            disable=not verbose,
        ):
            pred = (
                estimator.predict(data[val_indices])
                if self.metadata["task"] == "regression"
                else estimator.predict_proba(data[val_indices])
            )
            if oofp is None:
                oofp = np.full((len(data),) + pred.shape[1:], np.nan, dtype=np.float32)
            oofp[val_indices] = pred.astype(np.float32)
        assert oofp is not None
        if self.metadata["task"] == "binary":
            oofp = np.stack([1 - oofp, oofp], axis=-1)
            assert oofp is not None
        return oofp

    def infers_to_labels(self, pred: npt.NDArray[np.float32]) -> npt.NDArray:
        """推論結果(infer, infer_oof)からクラス名などを返す。

        Args:
            pred: 推論結果

        Returns:
            クラス名など

        """
        assert self.metadata["task"] in ("binary", "multiclass")
        assert self.metadata["class_names"] is not None
        class_names = np.array(self.metadata["class_names"])
        assert pred.shape == (len(pred), len(class_names))
        return class_names[np.argmax(pred, axis=-1)]


def train(
    data: pd.DataFrame | pl.DataFrame,
    labels: npt.ArrayLike,
    groups: npt.ArrayLike | None = None,
    folds: typing.Sequence[tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]]
    | None = None,
    task: typing.Literal["classification", "regression", "auto"] = "auto",
    estimator: sklearn.base.BaseEstimator | None = None,
) -> Model:
    """学習

    Args:
        data: 入力データ
        labels: ラベル
        groups: 入力データのグループ
        folds: CVの分割情報
        do_early_stopping: Early Stoppingをするのか否か
        hpo: optunaによるハイパーパラメータ探索を行うのか否か
        do_bagging: bagging_fraction, feature_fractionを設定するのか否か。
                    ラウンド数が少ない場合はFalseの方が安定するかも。
                    hpo=Trueなら効果なし。
        categorical_feature: カテゴリ列
        encode_categoricals: categorical_featureに指定した列をエンコードするか否か


    Returns:
        モデル

    """
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()
    columns = data.columns.to_list()
    data = data.to_numpy()
    labels = np.asarray(labels)
    assert len(data) == len(labels), f"{len(data)=} {len(labels)=}"

    class_names: list[typing.Any] | None = None
    task_: typing.Literal["binary", "multiclass", "regression"]
    if task == "classification" or isinstance(labels[0], str):
        # 分類の場合
        class_names = np.sort(np.unique(labels)).tolist()
        assert class_names is not None
        labels = _class_to_index(labels, class_names)
        assert len(class_names) >= 2
        if len(class_names) == 2:
            # 2クラス分類
            task_ = "binary"
        else:
            # 多クラス分類
            task_ = "multiclass"
    else:
        # 回帰の場合
        assert labels.dtype.type is np.float32
        task_ = "regression"

    if estimator is None:
        if task_ == "regression":
            estimator = sklearn.ensemble.RandomForestRegressor(random_state=1)
        else:
            estimator = sklearn.ensemble.RandomForestClassifier(random_state=1)

    if folds is None:
        if task_ == "regression":
            folds = list(sklearn.model_selection.KFold().split(data))
        else:
            folds = list(sklearn.model_selection.StratifiedKFold().split(data, labels))

    scores = sklearn.model_selection.cross_validate(
        estimator,
        data,
        labels,
        groups=groups,
        cv=folds,
        verbose=10,
        return_train_score=True,
        return_estimator=True,
    )

    # ログ出力
    logger.info(
        "sklearn:"
        f" train_score={scores['train_score'].mean():.3f}"
        f" val_score={scores['test_score'].mean():.3f}"
    )

    model = Model(
        scores["estimator"],
        ModelMetadata(
            task=task_,
            columns=columns,
            class_names=class_names,
            nfold=len(scores["estimator"]),
        ),
    )

    return model


def load(model_dir: str | os.PathLike[str]) -> Model:
    """学習済みモデルの読み込み

    Args:
        model_dir: 保存先ディレクトリ

    Returns:
        モデル

    """
    return Model.load(model_dir)


def _class_to_index(
    labels: npt.ArrayLike, class_names: list[typing.Any]
) -> npt.NDArray[np.int32]:
    return np.vectorize(class_names.index)(labels)
