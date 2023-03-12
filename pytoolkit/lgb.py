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
        score = model.evaluate(test_data, test_labels)
        assert 0.0 <= score <= 1.0

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

# だいぶお行儀が悪いけどimportされた時点でlightgbmにロガーを登録しちゃう
logger = logging.getLogger(__name__)
logger.addFilter(
    lambda r: r.getMessage()
    != "[LightGBM] [Warning] No further splits with positive gain, best gain: -inf"
)
lgb.register_logger(logger)


class Model:
    """テーブルデータのモデル。"""

    def __init__(self, boosters: list[lgb.Booster], metadata: dict[str, typing.Any]):
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
        metadata: dict[str, typing.Any] = json.loads(
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
            labels = _class_to_index(labels, self.metadata["class_names"])
            return {
                "acc": float(sklearn.metrics.accuracy_score(labels, np.round(pred))),
                "auc": float(sklearn.metrics.roc_auc_score(labels, pred)),
            }
        elif self.metadata["task"] == "multiclass":
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
        for c, values in self.metadata["encode_categoricals"].items():
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
        for c, values in self.metadata["encode_categoricals"].items():
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


def load_labeled_data(
    data_path: str | os.PathLike[str], label_col_name: str
) -> tuple[pl.DataFrame, npt.NDArray]:
    """ラベルありデータの読み込み

    Args:
        data_path: データのパス(CSV, Excelなど)
        label_col_name: ラベルの列名

    Returns:
        データフレーム

    """
    data = load_unlabeled_data(data_path)
    labels = data.drop_in_place(label_col_name).to_numpy()
    return data, labels


def load_unlabeled_data(data_path: str | os.PathLike[str]) -> pl.DataFrame:
    """ラベルなしデータの読み込み

    Args:
        data_path: データのパス(CSV, Excelなど)

    Returns:
        データフレーム

    """
    data_path = pathlib.Path(data_path)
    data: pl.DataFrame
    if data_path.suffix.lower() == ".csv":
        data = pl.read_csv(data_path)
    elif data_path.suffix.lower() == ".tsv":
        data = pl.read_csv(data_path, sep="\t")
    elif data_path.suffix.lower() == ".arrow":
        data = pl.read_ipc(data_path)
    elif data_path.suffix.lower() == ".parquet":
        data = pl.read_parquet(data_path)
    elif data_path.suffix.lower() in (".xls", ".xlsx", ".xlsm"):
        data = pl.read_excel(data_path)  # type: ignore
    else:
        raise ValueError(f"Unknown suffix: {data_path}")
    return data


def train(
    data: pd.DataFrame | pl.DataFrame,
    labels: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    groups: npt.ArrayLike | None = None,
    folds: typing.Sequence[tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]]
    | None = None,
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
    }
    metadata: dict[str, typing.Any] = {}

    if isinstance(labels[0], str):
        # 分類の場合
        class_names: list[str] = np.sort(np.unique(labels)).tolist()
        metadata["class_names"] = class_names
        labels = _class_to_index(labels, class_names)
        assert len(class_names) >= 2
        if len(class_names) == 2:
            # 2クラス分類
            metadata["task"] = "binary"
            if params.get("metric") is None:
                params["metric"] = ["auc", "binary_error"]
        else:
            # 多クラス分類
            metadata["task"] = "multiclass"
            params["num_class"] = len(class_names)
            if params.get("metric") is None:
                params["metric"] = ["multi_logloss", "multi_error"]
    else:
        # 回帰の場合
        assert labels.dtype.type is np.float32
        metadata["task"] = "regression"
        if params.get("metric") is None:
            params["metric"] = ["l2", "mae", "rmse"]
    params["objective"] = objective or metadata["task"]

    # カテゴリ列のエンコード
    metadata["encode_categoricals"] = {}
    if encode_categoricals and isinstance(categorical_feature, list):
        for c in categorical_feature:
            values = np.sort(data[c].dropna().unique()).tolist()
            logger.info(f"lgb: encode_categoricals({c}) => {values}")
            metadata["encode_categoricals"][c] = values
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
            seed=1,
            optuna_seed=1,
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
        seed=1,
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
    metadata["params"] = params

    # ログ出力
    cvbooster = typing.cast(lgb.CVBooster, cv_result["cvbooster"])
    logger.info(f"lgb: best_iteration={cvbooster.best_iteration}")
    for k in cv_result:
        if k != "cvbooster":
            score = float(cv_result[k][-1])
            logger.info(f"lgb: {k}={score:.3f}")
            metadata[k] = score
    metadata["best_iteration"] = cvbooster.best_iteration
    metadata["nfold"] = len(cvbooster.boosters)

    model = Model(cvbooster.boosters, metadata)

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
    labels: npt.ArrayLike, class_names: list[str]
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