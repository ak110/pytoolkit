"""LightGBM関連。

importしただけでlightgbm.register_loggerを呼び出すため注意。

Examples:
    ::

        import pytoolkit.lgb
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
        model = pytoolkit.lgb.train(train_data, train_labels, groups=None)

        # 保存・読み込み
        model.save(model_dir)
        model = pytoolkit.lgb.load(model_dir)

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

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import psutil
import sklearn.metrics
import tqdm

import pytoolkit.base

# だいぶお行儀が悪いけどimportされた時点でlightgbmにロガーを登録しちゃう
logger = logging.getLogger(__name__)
logger.addFilter(
    lambda r: r.getMessage()
    != "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf"
)
lgb.register_logger(logger)


class ModelMetadata(typing.TypedDict):
    """モデルのメタデータの型定義。"""

    task: typing.Literal["binary", "multiclass", "regression"]
    categorical_values: dict[str, list[typing.Any]]
    class_names: list[typing.Any] | None
    params: dict[str, typing.Any]
    nfold: int
    best_iteration: int
    cv_scores: dict[str, float]


class Model(pytoolkit.base.BaseModel):
    """テーブルデータのモデル。"""

    def __init__(self, boosters: list[lgb.Booster], metadata: ModelMetadata):
        self.boosters = boosters
        self.metadata = metadata

    def save(self, model_dir: str | os.PathLike[str]) -> None:
        """保存。

        Args:
            model_dir: 保存先ディレクトリ

        """
        model_dir = pathlib.Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        for fold, booster in enumerate(self.boosters):
            booster.save_model(
                str(model_dir / f"model.fold{fold}.txt"),
                num_iteration=self.metadata["best_iteration"],
            )

        df_importance = pl.DataFrame(
            {
                "feature": self.get_feature_names(),
                "importance[%]": self.get_feature_importance(normalize=True) * 100,
            }
        )
        df_importance = df_importance.sort("importance[%]", descending=True)
        df_importance.write_csv(model_dir / "feature_importance.csv")

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
        boosters: list[lgb.Booster] = []
        for fold in range(metadata["nfold"]):
            boosters.append(
                lgb.Booster(model_file=str(model_dir / f"model.fold{fold}.txt"))
            )
        return cls(boosters, metadata)

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
        data = data[self.get_feature_names()]
        for c, values in self.metadata["categorical_values"].items():
            data[c] = data[c].map(values.index, na_action="ignore")

        pred = np.mean(
            [
                booster.predict(data, num_iteration=self.metadata["best_iteration"])
                for booster in tqdm.tqdm(
                    self.boosters,
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
        assert len(folds) == len(self.boosters)
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        data = data[self.get_feature_names()]
        for c, values in self.metadata["categorical_values"].items():
            data[c] = data[c].map(values.index, na_action="ignore")

        oofp: npt.NDArray[np.float32] | None = None
        for booster, (_, val_indices) in tqdm.tqdm(
            list(zip(self.boosters, folds, strict=True)),
            ascii=True,
            ncols=100,
            desc="infer_oof",
            disable=not verbose,
        ):
            pred = booster.predict(
                data.iloc[val_indices], num_iteration=self.metadata["best_iteration"]
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

    def get_feature_names(self) -> list[str]:
        """列名を返す。

        Returns:
            列名の配列

        """
        return self.boosters[0].feature_name()

    def get_feature_importance(
        self, importance_type="gain", normalize: bool = True
    ) -> npt.NDArray[np.float32]:
        """feature importanceを返す。

        Args:
            importance_type: "split" or "gain"
            normalize: Trueの場合、合計が1になる配列を返す

        Returns:
            feature importance

        """
        feature_importance = np.mean(
            [
                gbm.feature_importance(importance_type=importance_type)
                for gbm in self.boosters
            ],
            axis=0,
            dtype=np.float32,
        )
        if normalize:
            feature_importance /= feature_importance.sum() + 1e-7
        return feature_importance


def train(
    data: pd.DataFrame | pl.DataFrame,
    labels: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    groups: npt.ArrayLike | None = None,
    folds: typing.Sequence[tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]]
    | None = None,
    task: typing.Literal["classification", "regression", "auto"] = "auto",
    categorical_feature: typing.Literal["auto"] | list[str] = "auto",
    init_score: npt.ArrayLike | None = None,
    num_boost_round: int = 9999,
    do_early_stopping: bool = True,
    first_metric_only: bool = True,
    learning_rate: float = 0.1,
    objective=None,
    metric=None,
    fobj=None,
    feval=None,
    eval_train_metric: bool = False,
    hpo: bool = False,
    do_bagging: bool = True,
    encode_categoricals: bool = True,
    seed: int = 1,
) -> Model:
    """学習

    Args:
        data: 入力データ
        labels: ラベル
        weights: 入力データの重み
        groups: 入力データのグループ
        folds: CVの分割情報
        do_early_stopping: Early Stoppingをするのか否か
        hpo: optunaによるハイパーパラメータ探索を行うのか否か
        do_bagging: bagging_fraction, feature_fractionを設定するのか否か。
                    ラウンド数が少ない場合はFalseの方が安定するかも。
                    hpo=Trueなら効果なし。
        categorical_feature: カテゴリ列
        encode_categoricals: categorical_featureに指定した列をエンコードするか否か
        seed: 乱数シード

    Returns:
        モデル

    """
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()
    labels = np.asarray(labels)
    assert len(data) == len(labels), f"{len(data)=} {len(labels)=}"

    params: dict[str, typing.Any] = {
        "learning_rate": learning_rate,
        "metric": metric,
        "nthread": psutil.cpu_count(logical=False),
        "force_col_wise": True,
        "data_random_seed": seed + 0,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
    }

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
            if params.get("metric") is None:
                params["metric"] = ["auc", "binary_error"]
        else:
            # 多クラス分類
            task_ = "multiclass"
            params["num_class"] = len(class_names)
            if params.get("metric") is None:
                params["metric"] = ["multi_logloss", "multi_error"]
    else:
        # 回帰の場合
        assert labels.dtype.type is np.float32
        task_ = "regression"
        if params.get("metric") is None:
            params["metric"] = ["l2", "mae", "rmse"]
    params["objective"] = objective or task_

    # カテゴリ列のエンコード
    categorical_values: dict[str, list[typing.Any]] = {}
    if encode_categoricals and isinstance(categorical_feature, list):
        for c in categorical_feature:
            values = np.sort(data[c].dropna().unique()).tolist()
            logger.info(f"lgb: encode_categoricals({c}) => {values}")
            categorical_values[c] = values
            data[c] = data[c].map(values.index, na_action="ignore")

    train_set = lgb.Dataset(
        data,
        labels,
        weight=weights,
        group=groups,
        init_score=init_score,
        categorical_feature=categorical_feature,  # type: ignore
        free_raw_data=False,
    )

    if hpo:
        import optuna

        logger.info(f"HPO開始: {learning_rate=}")
        params["deterministic"] = True
        params["verbosity"] = -1
        tuner = optuna.integration.lightgbm.LightGBMTunerCV(
            params,
            train_set,
            folds=folds,
            fobj=fobj,
            feval=feval,
            categorical_feature=categorical_feature,  # type: ignore
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(
                    min(max(int(10 / params["learning_rate"] ** 0.5), 20), 200),
                    first_metric_only=first_metric_only,
                    verbose=False,
                )
            ]
            if do_early_stopping
            else [],
            seed=seed,
            optuna_seed=seed,
        )
        tuner.run()
        params.update(tuner.best_params)
        params["learning_rate"] /= 10.0
        params.pop("verbosity")
        params.pop("deterministic")
        logger.info(f"HPO完了: {tuner.best_score=:.3f}")
    elif do_bagging:
        params["bagging_freq"] = 1
        params["bagging_fraction"] = 0.8
        params["feature_fraction"] = 0.8

    logger.info(f"lgb: {params=}")
    # 0.1のとき10、0.01のとき100くらいにしてみる
    cv_result = lgb.cv(
        params,
        train_set,
        folds=folds,
        fobj=fobj,
        feval=feval,
        categorical_feature=categorical_feature,  # type: ignore
        num_boost_round=num_boost_round,
        verbose_eval=None,
        seed=seed,
        callbacks=(
            [
                lgb.early_stopping(
                    min(max(int(10 / params["learning_rate"] ** 0.5), 20), 200),
                    first_metric_only=first_metric_only,
                )
            ]
            if do_early_stopping
            else []
        )
        + [EvaluationLogger()],
        eval_train_metric=eval_train_metric,
        return_cvbooster=True,
    )

    # ログ出力
    cvbooster = typing.cast(lgb.CVBooster, cv_result["cvbooster"])
    logger.info(f"lgb: best_iteration={cvbooster.best_iteration}")
    cv_scores: dict[str, float] = {}
    for k in cv_result:
        if k != "cvbooster":
            score = float(cv_result[k][-1])
            logger.info(f"lgb: {k}={score:.3f}")
            cv_scores[k] = score

    model = Model(
        cvbooster.boosters,
        ModelMetadata(
            task=task_,
            categorical_values=categorical_values,
            class_names=class_names,
            params=params,
            nfold=len(cvbooster.boosters),
            best_iteration=cvbooster.best_iteration,
            cv_scores=cv_scores,
        ),
    )

    # feature importance
    feature_names = model.get_feature_names()
    fi = model.get_feature_importance(normalize=True)
    logger.info("feature importance:")
    for i in (-fi).argsort()[: min(len(fi), 10)]:
        logger.info(f"  {feature_names[i]}: {fi[i]:.1%}")

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


class EvaluationLogger:
    """pythonのloggingモジュールでログ出力するcallback。

    References:
        - <https://amalog.hateblo.jp/entry/lightgbm-logging-callback>

    """

    def __init__(self, period=100, show_stdv=True, level=logging.INFO):
        self.period = period
        self.show_stdv = show_stdv
        self.level = level
        self.order = 10

    def __call__(self, env):
        if env.evaluation_result_list:
            # 最初だけ1, 2, 4, ... で出力。それ以降はperiod毎。
            n = env.iteration + 1
            if n < self.period:
                t = (n & (n - 1)) == 0
            else:
                t = n % self.period == 0
            if t:
                result = "  ".join(
                    [
                        _format_eval_result(x, show_stdv=self.show_stdv)
                        for x in env.evaluation_result_list
                    ]
                )
                logger.log(self.level, f"[{n:4d}]  {result}")


def _format_eval_result(value, show_stdv=True):
    """Format metric string."""
    if len(value) == 4:
        return f"{value[0]}'s {value[1]}: {value[2]:.4f}"
    if len(value) == 5:
        if show_stdv:
            return f"{value[0]}'s {value[1]}: {value[2]:.4f} + {value[4]:.4f}"
        return f"{value[0]}'s {value[1]}: {value[2]:.4f}"
    raise ValueError(f"Wrong metric value: {value}")
