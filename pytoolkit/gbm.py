"""LightGBMなどなど関連。"""
import pathlib

import numpy as np

import pytoolkit as tk


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
    """LightGBMのモデル複数をまとめて扱うクラス。

    Args:
        gbms (array-like): shape=(num_folds, num_seeds)の配列
        folds (array-like): CV
        seeds (array-like): seed ensemble用のseedの配列

    """

    def __init__(self, gbms, folds=None, seeds=None):
        self.gbms = gbms
        self.folds = folds
        self.seeds = seeds

    def save(self, models_dir):
        """保存。"""
        models_dir = pathlib.Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        for fold, gbms_fold in enumerate(self.gbms):
            for seed, gbm in zip(self.seeds, gbms_fold):
                gbm.save_model(
                    str(models_dir / f"model.fold{fold}.seed{seed}.txt"),
                    num_iteration=gbm.best_iteration,
                )
        # ついでにfeature_importanceも。
        df_importance = self.feature_importance()
        df_importance.to_excel(str(models_dir / "feature_importance.xlsx"))

    @classmethod
    def load(cls, models_dir, *, num_folds=None, folds=None, seeds=None):
        """読み込み。"""
        assert num_folds is not None or folds is not None
        import lightgbm as lgb

        seeds = [123] if seeds is None else seeds
        gbms = np.array(
            [
                [
                    lgb.Booster(
                        model_file=str(models_dir / f"model.fold{fold}.seed{seed}.txt")
                    )
                    for seed in seeds
                ]
                for fold in range(num_folds or len(folds))
            ]
        )
        return cls(gbms=gbms, folds=folds, seeds=seeds)

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
        seeds=None,
        **kwargs,
    ):
        """LightGBMのcv関数を呼び出す。

        Args:
            seeds (array-like): seed ensemble用。

        Returns:
            LGBModels: モデル。

        """
        import lightgbm as lgb

        seeds = [123] if seeds is None else seeds

        gbms = np.empty((len(folds), len(seeds)), dtype=object)
        for seed_i, seed in enumerate(seeds):
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
                **kwargs,
            )
            for k in eval_hist:
                if k.endswith("-mean"):
                    tk.log.get(__name__).info(
                        f"cv {k[:-5]}: {np.float32(eval_hist[k][-1])}"
                    )
            gbms[:, seed_i] = model_extractor.raw_boosters
            # 怪しいけどとりあえずいったん書き換えちゃう
            for gbm in gbms[:, seed_i]:
                gbm.best_iteration = model_extractor.best_iteration

        return cls(gbms=gbms, folds=folds, seeds=seeds)

    def predict_oof(self, X, reduce=True):
        """out-of-foldなpredict結果を返す。

        Args:
            X (pd.DataFrame): 入力データ
            reduce (bool): seed ensemble分を平均して返すならTrue、配列のまま返すならFalse

        Returns:
            np.ndarray: 予測結果

        """
        assert self.folds is not None
        oofp_list = []
        for gbms_fold, (_, val_indices) in zip(self.gbms, self.folds):
            pred_val = [
                gbm.predict(
                    X.iloc[val_indices][gbm.feature_name()],
                    num_iteration=gbm.best_iteration,
                )
                for gbm in gbms_fold
            ]
            if reduce:
                pred_val = np.mean(pred_val, axis=0)
            else:
                pred_val = np.transpose(pred_val, axes=(1, 0))
            oofp_list.append((val_indices, pred_val))

        if reduce:
            oofp_shape = (len(X),) + oofp_list[0][1].shape[1:]
            oofp = np.zeros(oofp_shape, dtype=oofp_list[0][1].dtype)
        else:
            oofp_shape = (len(X), len(self.gbms[0])) + oofp_list[0][1][0].shape[1:]
            oofp = np.zeros(oofp_shape, dtype=oofp_list[0][1][0].dtype)
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
            np.mean(
                [
                    gbm.predict(X[gbm.feature_name()], num_iteration=gbm.best_iteration)
                    for gbm in gbms_fold
                ],
                axis=0,
            )
            for gbms_fold in self.gbms
        ]

    def feature_importance(self, importance_type="gain"):
        """Feature ImportanceをDataFrameで返す。"""
        import pandas as pd

        columns = self.gbms[0][0].feature_name()
        for gbms_fold in self.gbms:
            for gbm in gbms_fold:
                assert tuple(columns) == tuple(gbm.feature_name())

        t = np.int32 if importance_type == "split" else np.float32
        fi = np.zeros((len(columns),), dtype=t)
        for gbms_fold in self.gbms:
            for gbm in gbms_fold:
                fi += gbm.feature_importance(importance_type=importance_type)

        return pd.DataFrame(data={"importance": fi}, index=columns)


class CBModels:
    """CatBoostのモデル複数をまとめて扱うクラス。"""

    def __init__(self, gbms, folds=None, train_pool=None):
        self.gbms = gbms
        self.folds = folds
        self.train_pool = train_pool

    def save(self, models_dir):
        """保存。"""
        assert self.train_pool is not None
        models_dir = pathlib.Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        for fold, gbm in enumerate(self.gbms):
            gbm.save_model(
                str(models_dir / f"model.fold{fold}.cbm"), pool=self.train_pool
            )
        # ついでにfeature_importanceも。
        df_importance = self.feature_importance()
        df_importance.to_excel(str(models_dir / "feature_importance.xlsx"))

    @classmethod
    def load(cls, models_dir, *, num_folds=None, folds=None):
        """読み込み。"""
        assert num_folds is not None or folds is not None
        import catboost

        def load(model_path):
            gbm = catboost.CatBoost()
            gbm.load_model(model_path)
            return gbm

        gbms = [
            load(str(models_dir / f"model.fold{fold}.cbm"))
            for fold in range(num_folds or len(folds))
        ]
        return cls(gbms=gbms, folds=folds)

    @classmethod
    def cv(cls, params, train_pool, *, folds, **kwargs):
        """CatBoostでCV。

        Returns:
            CBModels: モデル。

        """
        import catboost

        gbms, score_list = [], []
        for fold, (train_indices, val_indices) in enumerate(folds):
            with tk.log.trace_scope(f"cv({fold + 1}/{len(folds)})"):
                gbm = catboost.train(
                    params=params,
                    pool=train_pool.slice(train_indices),
                    eval_set=train_pool.slice(val_indices),
                    **kwargs,
                )
                gbms.append(gbm)
                score_list.append(gbm.get_best_score()["validation"])

        cv_weights = [len(val_indices) for _, val_indices in folds]
        for k in score_list[0]:
            score = [s[k] for s in score_list]
            score = np.average(score, weights=cv_weights)
            tk.log.get(__name__).info(f"cv {k}: {np.float32(score)}")

        return cls(gbms=gbms, folds=folds, train_pool=train_pool)

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
            pred_val = gbm.predict(X.iloc[val_indices])
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
        return [gbm.predict(X) for gbm in self.gbms]

    def feature_importance(self):
        """Feature ImportanceをDataFrameで返す。"""
        import pandas as pd

        columns = self.gbms[0].feature_names_
        for gbm in self.gbms:
            assert tuple(columns) == tuple(gbm.feature_names_)

        fi = np.zeros((len(columns),), dtype=np.float32)
        for gbm in self.gbms:
            fi += gbm.get_feature_importance()

        return pd.DataFrame(data={"importance": fi}, index=columns)


class XGBModels:
    """XGBoostのモデル複数をまとめて扱うクラス。"""

    def __init__(self, gbms, folds=None):
        self.gbms = gbms
        self.folds = folds

    def save(self, models_dir):
        """保存。"""
        models_dir = pathlib.Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        for fold, gbm in enumerate(self.gbms):
            tk.utils.dump(gbm, models_dir / f"model.fold{fold}.pkl")
        # ついでにfeature_importanceも。
        df_importance = self.feature_importance()
        df_importance.to_excel(str(models_dir / "feature_importance.xlsx"))

    @classmethod
    def load(cls, models_dir, *, num_folds=None, folds=None):
        """読み込み。"""
        assert num_folds is not None or folds is not None
        gbms = [
            tk.utils.load(models_dir / f"model.fold{fold}.pkl")
            for fold in range(num_folds or len(folds))
        ]
        return cls(gbms=gbms, folds=folds)

    @classmethod
    def cv(
        cls,
        params,
        train_set,
        early_stopping_rounds=200,
        num_boost_round=9999,
        verbose_eval=100,
        callbacks=None,
        *,
        folds,
        **kwargs,
    ):
        """XGBoostでCV。

        Returns:
            XGBModels: モデル。

        """
        import xgboost as xgb

        gbms = []

        def model_extractor(env):
            gbms.clear()
            gbms.extend([f.bst for f in env.cvfolds])

        eval_hist = xgb.cv(
            params,
            dtrain=train_set,
            folds=folds,
            callbacks=(callbacks or []) + [model_extractor],
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            **kwargs,
        )
        for k, v in eval_hist.items():
            if k.endswith("-mean"):
                tk.log.get(__name__).info(f"{k[:-5]}: {v.values[-1]}")

        return cls(gbms=gbms, folds=folds)

    def predict_oof(self, X):
        """out-of-foldなpredict結果を返す。

        Args:
            X (pd.DataFrame): 入力データ

        Returns:
            np.ndarray: 予測結果

        """
        assert self.folds is not None
        import xgboost as xgb

        oofp_list = []
        for gbm, (_, val_indices) in zip(self.gbms, self.folds):
            data = xgb.DMatrix(data=X.iloc[val_indices], feature_names=X.columns.values)
            pred_val = gbm.predict(data)
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
        import xgboost as xgb

        data = xgb.DMatrix(data=X, feature_names=X.columns.values)
        return [gbm.predict(data) for gbm in self.gbms]

    def feature_importance(self, importance_type="gain"):
        """Feature ImportanceをDataFrameで返す。"""
        import pandas as pd

        columns = self.gbms[0].feature_names
        for gbm in self.gbms:
            assert tuple(columns) == tuple(gbm.feature_names)

        fi = np.zeros((len(columns),), dtype=np.float32)
        for gbm in self.gbms:
            d = gbm.get_score(importance_type=importance_type)
            fi += [d.get(c, 0) for c in columns]

        return pd.DataFrame(data={"importance": fi}, index=columns)
