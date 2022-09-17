"""LightGBM関連。"""
from __future__ import annotations

import logging
import pathlib
import typing

import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm

if typing.TYPE_CHECKING:
    import lightgbm as lgb

logger = logging.getLogger(__name__)


def cv(
    models_dir: str | pathlib.Path,
    feats_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    folds: typing.Sequence[tuple[np.ndarray, np.ndarray]],
    params: dict,
    weights: list | np.ndarray | pd.Series = None,
    groups: list | np.ndarray | pd.Series = None,
    init_score: list | np.ndarray | pd.Series = None,
    fobj=None,
    feval=None,
    early_stopping_rounds: int = 200,
    num_boost_round: int = 9999,
    verbose_eval: int = 100,
    importance_type: str = "gain",
) -> None:
    """LightGBMでCVしてできたモデルをmodels_dir配下に保存する。"""
    import lightgbm as lgb  # pylint: disable=redefined-outer-name

    # 学習
    train_set = lgb.Dataset(
        feats_train,
        y_train,
        weight=weights,
        group=groups,
        init_score=init_score,
        free_raw_data=False,
    )
    cv_result = lgb.cv(
        params,
        train_set,
        folds=folds,
        fobj=fobj,
        feval=feval,
        early_stopping_rounds=early_stopping_rounds,
        num_boost_round=num_boost_round,
        verbose_eval=None,
        callbacks=[EvaluationLogger(period=verbose_eval)],
        seed=1,
        return_cvbooster=True,
    )

    # ログ出力
    cvbooster = typing.cast(lgb.CVBooster, cv_result["cvbooster"])
    logger.info(f"lgb: best_iteration={cvbooster.best_iteration}")
    for k in cv_result:
        if k != "cvbooster":
            score = np.float32(cv_result[k][-1])
            logger.info(f"lgb: {k}={score:.3f}")

    # 保存
    models_dir = pathlib.Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for fold, booster in enumerate(cvbooster.boosters):
        booster.save_model(
            str(models_dir / f"model.fold{fold}.txt"),
            num_iteration=cvbooster.best_iteration,
        )

    columns = (
        feats_train.columns.values
        if isinstance(feats_train, pd.DataFrame)
        else [f"feature_{i}" for i in range(feats_train.shape[1])]
    )
    t = np.int32 if importance_type == "split" else np.float32
    fi: typing.Any = np.zeros((len(columns),), dtype=t)
    for gbm in typing.cast(typing.List[lgb.Booster], cvbooster.boosters):
        fi += gbm.feature_importance(importance_type=importance_type)
    df = pd.DataFrame(data={"importance": fi}, index=columns)
    df = df.sort_values(by="importance", ascending=False)
    df.to_csv(models_dir / "feature_importance.csv", index_label="feature")


def load(models_dir, nfold: int) -> list[lgb.Booster]:
    """cvで保存したモデルの読み込み。"""
    import lightgbm as lgb  # pylint: disable=redefined-outer-name

    models_dir = pathlib.Path(models_dir)
    return [
        lgb.Booster(model_file=str(models_dir / f"model.fold{fold}.txt"))
        for fold in range(nfold)
    ]


def predict(
    boosters: list[lgb.Booster], feats_test: pd.DataFrame | np.ndarray
) -> list[np.ndarray]:
    """推論。"""
    return [
        booster.predict(feats_test)
        for booster in tqdm.tqdm(boosters, ascii=True, ncols=100, desc="predict")
    ]


def predict_oof(
    boosters: list[lgb.Booster],
    feats_train: pd.DataFrame | np.ndarray,
    folds: typing.Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """out-of-fold predictions。"""
    oof = None
    for (_, val_indices), booster in tqdm.tqdm(
        list(zip(folds, boosters)), ascii=True, ncols=100, desc="predict_oof"
    ):
        pred = booster.predict(_gather(feats_train, val_indices))
        if oof is None:
            oof = np.zeros((len(feats_train),) + pred.shape[1:], dtype=pred.dtype)
        oof[val_indices] = pred
    assert oof is not None
    return oof


def _gather(feats: pd.DataFrame | np.ndarray, indices):
    if isinstance(feats, pd.DataFrame):
        return feats.iloc[indices]
    return feats[indices]


def f1_metric(preds, data):
    """LightGBM用2クラスF1スコア"""
    y_true = data.get_label()
    preds = np.round(preds)
    return "f1", sklearn.metrics.f1_score(y_true, preds), True


def mf1_metric(preds, data):
    """LightGBM用多クラスF1スコア(マクロ平均)"""
    y_true = data.get_label()
    num_classes = len(preds) // len(y_true)
    preds = preds.reshape(-1, num_classes).argmax(axis=-1)
    return "f1", sklearn.metrics.f1_score(y_true, preds, average="macro"), True


def r2_metric(preds, data):
    """LightGBM用R2"""
    y_true = data.get_label()
    return "r2", sklearn.metrics.r2_score(y_true, preds), True


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
    elif len(value) == 5:
        if show_stdv:
            return f"{value[0]}'s {value[1]}: {value[2]:.4f} + {value[4]:.4f}"
        else:
            return f"{value[0]}'s {value[1]}: {value[2]:.4f}"
    else:
        raise ValueError("Wrong metric value")
