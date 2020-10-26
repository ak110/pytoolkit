"""LightGBM関連。"""
from __future__ import annotations

import logging
import typing

import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:
    import lightgbm as lgb

logger = logging.getLogger(__name__)


def cv(
    feats_train: typing.Union[pd.DataFrame, np.ndarray],
    y_train: np.ndarray,
    folds,
    params: dict,
    weights: typing.Union[list, np.ndarray, pd.Series] = None,
    groups: typing.Union[list, np.ndarray, pd.Series] = None,
    init_score: typing.Union[list, np.ndarray, pd.Series] = None,
    fobj=None,
    feval=None,
    early_stopping_rounds: int = 200,
    num_boost_round: int = 9999,
    verbose_eval: int = 100,
) -> typing.Tuple[typing.List[lgb.Booster], int]:
    """LightGBMでCVしてboostersとbest_iterationを返す。"""
    import lightgbm as lgb  # pylint: disable=redefined-outer-name

    train_set = lgb.Dataset(
        feats_train,
        y_train,
        weight=weights,
        group=groups,
        init_score=init_score,
        free_raw_data=False,
    )
    model_extractor = ModelExtractionCallback()
    eval_hist = lgb.cv(
        params,
        train_set,
        folds=folds,
        fobj=fobj,
        feval=feval,
        early_stopping_rounds=early_stopping_rounds,
        num_boost_round=num_boost_round,
        verbose_eval=verbose_eval,
        callbacks=[model_extractor],
        seed=1,
    )
    best_iteration = model_extractor.best_iteration

    logger.info(f"lgb: {best_iteration=}")
    for k in eval_hist:
        score = np.float32(eval_hist[k][best_iteration - 1])
        logger.info(f"lgb: {k}={score:.3f}")

    return model_extractor.raw_boosters, model_extractor.best_iteration


def predict(
    boosters: typing.List[lgb.Booster],
    best_iteration: int,
    feats_test: typing.Union[pd.DataFrame, np.ndarray],
) -> typing.List[np.ndarray]:
    """推論。"""
    return [bst.predict(feats_test, num_iteration=best_iteration) for bst in boosters]


class ModelExtractionCallback:
    """lightgbm.cv() から学習済みモデルを取り出すためのコールバックに使うクラス

    NOTE: 非公開クラス '_CVBooster' に依存しているため将来的に動かなく恐れがある

    References:
        - <https://blog.amedama.jp/entry/lightgbm-cv-model>

    """

    def __init__(self):
        self._model = None

    def __call__(self, env):
        # _CVBooster の参照を保持する
        self._model = env.model

    @property
    def boosters_proxy(self):
        """Booster へのプロキシオブジェクトを返す。"""
        assert self._model is not None, "callback has not called yet"
        return self._model

    @property
    def raw_boosters(self):
        """Booster のリストを返す。"""
        assert self._model is not None, "callback has not called yet"
        return self._model.boosters

    @property
    def best_iteration(self):
        """Early stop したときの boosting round を返す。"""
        assert self._model is not None, "callback has not called yet"
        return self._model.best_iteration
