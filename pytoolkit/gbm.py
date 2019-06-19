"""LightGBMなどなど関連。"""
import pathlib

import numpy as np


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

    def _assert_called_cb(self):
        if self._model is None:
            # コールバックが呼ばれていないときは例外にする
            raise RuntimeError("callback has not called yet")

    @property
    def boosters_proxy(self):
        """Booster へのプロキシオブジェクトを返す。"""
        self._assert_called_cb()
        return self._model

    @property
    def raw_boosters(self):
        """Booster のリストを返す。"""
        self._assert_called_cb()
        return self._model.boosters

    @property
    def best_iteration(self):
        """Early stop したときの boosting round を返す。"""
        self._assert_called_cb()
        return self._model.best_iteration


class LGBModels:
    """LightGBMのモデル複数をまとめて扱うクラス。"""

    def __init__(self, gbms, folds=None):
        self.gbms = gbms
        self.folds = folds

    def save(self, models_dir):
        """保存。"""
        models_dir = pathlib.Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        for fold, gbm in enumerate(self.gbms):
            gbm.save_model(
                str(models_dir / f"model.fold{fold}.txt"),
                num_iteration=gbm.best_iteration,
            )

    @classmethod
    def load(cls, models_dir, *, num_folds=None, folds=None):
        """読み込み。"""
        assert num_folds is not None or folds is not None
        import lightgbm as lgb

        gbms = [
            lgb.Booster(model_file=str(models_dir / f"model.fold{fold}.txt"))
            for fold in range(num_folds or len(folds))
        ]
        return cls(gbms=gbms, folds=folds)

    @classmethod
    def cv(
        cls,
        params,
        train_set,
        folds,
        early_stopping_rounds=200,
        num_boost_round=9999,
        verbose_eval=100,
        callbacks=None,
        seed=123,
    ):
        """LightGBMのcv関数を呼び出す。

        Returns:
            tuple: LGBModelsのインスタンスとeval_hist。

        """
        import lightgbm as lgb

        model_extractor = ModelExtractionCallback()
        eval_hist = lgb.cv(
            params,
            train_set,
            folds=folds,
            early_stopping_rounds=early_stopping_rounds,
            num_boost_round=num_boost_round,
            verbose_eval=verbose_eval,
            callbacks=(callbacks or []) + [model_extractor],
            seed=seed,
        )
        gbms = model_extractor.raw_boosters

        return cls(gbms=gbms, folds=folds), eval_hist

    def predict_oof(self, X):
        """out-of-foldなpredict結果を返す。

        Args:
            X (pd.DataFrame): 入力データ

        Returns:
            np.ndarray: 予測結果

        """
        assert self.folds is not None
        oofp_list = []
        for gbm, (_, val_indices) in zip(self.gbms, self.folds):
            pred_val = gbm.predict(
                X.iloc[val_indices][gbm.feature_name()],
                num_iteration=gbm.best_iteration,
            )
            oofp_list.append((val_indices, pred_val))

        oofp_shape = (len(X),) + oofp_list[0][1].shape[1:]
        oofp = np.zeros(oofp_shape, dtype=oofp_list[0][1].dtype)
        for val_indices, pred_val in oofp_list:
            oofp[val_indices] = pred_val

        return oofp

    def predict(self, X):
        """予測結果をリストで返す。

        Args:
            X (pd.DataFrame): 入力データ

        Returns:
            list: len(self.folds)個の予測結果

        """
        return [
            gbm.predict(X[gbm.feature_name()], num_iteration=gbm.best_iteration)
            for gbm in self.gbms
        ]

    def feature_importance(self, importance_type="split"):
        """Feature ImportanceをDataFrameで返す。"""
        import pandas as pd

        columns = self.gbms[0].feature_name()
        for gbm in self.gbms:
            assert tuple(columns) == tuple(gbm.feature_name())

        t = np.int32 if importance_type == "split" else np.float32
        fi = np.zeros((len(columns),), dtype=t)
        for gbm in self.gbms:
            fi += gbm.feature_importance(importance_type=importance_type)

        return pd.DataFrame(data={"importance": fi}, index=columns)
