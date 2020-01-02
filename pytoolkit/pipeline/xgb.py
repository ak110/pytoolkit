"""xgboost"""
from __future__ import annotations

import pathlib
import typing

import numpy as np
import pandas as pd
import sklearn.metrics

import pytoolkit as tk

from .core import Model


class XGBModel(Model):
    """XGBoostのモデル。

    Args:
        params: XGBoostのパラメータ
        nfold: cvの分割数
        models_dir: 保存先ディレクトリ
        early_stopping_rounds: xgboost.cvのパラメータ
        num_boost_round: xgboost.cvのパラメータ
        verbose_eval: xgboost.cvのパラメータ
        callbacks: xgboost.cvのパラメータ
        cv_params: xgboost.cvのパラメータ (kwargs)

    """

    def __init__(
        self,
        params: dict,
        nfold: int,
        models_dir: tk.typing.PathLike,
        early_stopping_rounds: int = 200,
        num_boost_round: int = 9999,
        verbose_eval: int = 100,
        callbacks: list = None,
        cv_params: dict = None,
        preprocessors: tk.pipeline.EstimatorListType = None,
        postprocessors: tk.pipeline.EstimatorListType = None,
    ):
        import xgboost

        super().__init__(nfold, models_dir, preprocessors, postprocessors)
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.verbose_eval = verbose_eval
        self.callbacks = callbacks
        self.cv_params = cv_params
        self.gbms_: typing.Optional[typing.List[xgboost.Booster]] = None
        self.best_ntree_limit_: typing.Optional[int] = None

    def _save(self, models_dir: pathlib.Path):
        assert self.gbms_ is not None
        tk.utils.dump(self.gbms_, models_dir / f"model.pkl")
        tk.utils.dump(self.best_ntree_limit_, models_dir / f"best_ntree_limit.pkl")
        # ついでにfeature_importanceも。
        df_importance = self.feature_importance()
        df_importance.to_excel(str(models_dir / "feature_importance.xlsx"))

    def _load(self, models_dir: pathlib.Path):
        self.gbms_ = tk.utils.load(models_dir / f"model.pkl")
        self.best_ntree_limit_ = tk.utils.load(models_dir / f"best_ntree_limit.pkl")
        assert self.gbms_ is not None
        assert len(self.gbms_) == self.nfold

    def _cv(
        self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType
    ) -> tk.evaluations.EvalsType:
        import xgboost

        assert isinstance(dataset.data, pd.DataFrame)

        train_set = xgboost.DMatrix(
            data=dataset.data,
            label=dataset.labels,
            weight=dataset.weights,
            feature_names=dataset.data.columns.values,
        )

        self.gbms_ = []

        def model_extractor(env):
            self.gbms_.clear()
            self.gbms_.extend([f.bst for f in env.cvfolds])

        eval_hist = xgboost.cv(
            self.params,
            dtrain=train_set,
            folds=folds,
            callbacks=(self.callbacks or []) + [model_extractor],
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval,
            **(self.cv_params or {}),
        )
        scores = {}
        for k, v in eval_hist.items():
            if k.endswith("-mean"):
                name, score = k[:-5], v.values[-1]
                scores[name] = score
                tk.log.get(__name__).info(f"{name}: {score}")
            self.best_ntree_limit_ = len(v)

        return scores

    def _predict(self, dataset: tk.data.Dataset, fold: int) -> np.ndarray:
        import xgboost

        assert self.gbms_ is not None
        assert self.best_ntree_limit_ is not None
        assert isinstance(dataset.data, pd.DataFrame)

        data = xgboost.DMatrix(
            data=dataset.data, feature_names=dataset.data.columns.values
        )
        gbm = self.gbms_[fold]
        return gbm.predict(data, ntree_limit=gbm.best_ntree_limit)

    def feature_importance(self, importance_type: str = "total_gain"):
        """Feature ImportanceをDataFrameで返す。"""
        assert self.gbms_ is not None

        columns = self.gbms_[0].feature_names
        for gbm in self.gbms_:
            assert tuple(columns) == tuple(gbm.feature_names)

        fi = np.zeros((len(columns),), dtype=np.float32)
        for gbm in self.gbms_:
            d = gbm.get_score(importance_type=importance_type)
            fi += [d.get(c, 0) for c in columns]

        return pd.DataFrame(data={"importance": fi}, index=columns)


def xgb_r2(preds, dtrain):
    """XGBoost用R2"""
    labels = dtrain.get_label()
    return "r2", np.float32(sklearn.metrics.r2_score(labels, preds))
