"""LightGBMなどなど関連。"""
import pathlib

import numpy as np

import pytoolkit as tk

from . import pipeline


class LGBModel(pipeline.Model):
    """LightGBMのモデル。

    Args:
        params (dict): lgb.cvのパラメータ
        nfold (int): cvの分割数
        early_stopping_rounds (int): lgb.cvのパラメータ
        num_boost_round (int): lgb.cvのパラメータ
        verbose_eval (int): lgb.cvのパラメータ
        callbacks (array-like): lgb.cvのパラメータ
        cv_params (dict): lgb.cvのパラメータ (**kwargs)
        seeds (array-like): seed ensemble用のseedの配列

    """

    def __init__(
        self,
        params,
        nfold,
        early_stopping_rounds=200,
        num_boost_round=9999,
        verbose_eval=100,
        callbacks=None,
        cv_params=None,
        seeds=None,
    ):
        self.params = params
        self.nfold = nfold
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.verbose_eval = verbose_eval
        self.callbacks = callbacks
        self.cv_params = cv_params
        self.seeds = seeds
        self.gbms_ = None

    def cv(self, dataset, folds, models_dir):
        """CVして保存。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex
            models_dir (pathlib.Path): 保存先ディレクトリ (Noneなら保存しない)

        Returns:
            dict: metrics名と値

        """
        import lightgbm as lgb

        # 独自拡張: sklearn風の指定
        if self.params.get("feature_fraction") == "sqrt":
            n = len(dataset.data.columns)
            self.params["feature_fraction"] = np.sqrt(n) / n
        elif self.params.get("feature_fraction") == "log2":
            n = len(dataset.data.columns)
            self.params["feature_fraction"] = np.log2(n) / n

        train_set = lgb.Dataset(
            dataset.data,
            dataset.labels,
            weight=dataset.weights if dataset.weights is not None else None,
            group=np.bincount(dataset.groups) if dataset.groups is not None else None,
            free_raw_data=False,
        )

        seeds = [123] if self.seeds is None else self.seeds

        scores = {}
        self.gbms_ = np.empty((len(folds), len(seeds)), dtype=object)
        for seed_i, seed in enumerate(seeds):
            with tk.log.trace_scope(f"seed averaging({seed_i + 1}/{len(seeds)})"):
                model_extractor = ModelExtractionCallback()
                eval_hist = lgb.cv(
                    self.params,
                    train_set,
                    folds=folds,
                    early_stopping_rounds=self.early_stopping_rounds,
                    num_boost_round=self.num_boost_round,
                    verbose_eval=self.verbose_eval,
                    callbacks=(self.callbacks or []) + [model_extractor],
                    seed=seed,
                    **(self.cv_params or {}),
                )
                tk.log.get(__name__).info(
                    f"best iteration: {model_extractor.best_iteration}"
                )
                for k in eval_hist:
                    if k.endswith("-mean"):
                        name, score = k[:-5], np.float32(eval_hist[k][-1])
                        if name not in scores:
                            scores[name] = []
                        scores[name].append(score)
                        tk.log.get(__name__).info(f"cv {name}: {score}")
                self.gbms_[:, seed_i] = model_extractor.raw_boosters
                # 怪しいけどとりあえずいったん書き換えちゃう
                for gbm in self.gbms_[:, seed_i]:
                    gbm.best_iteration = model_extractor.best_iteration

        if models_dir is not None:
            self.save(models_dir)

        for name, score_list in scores.items():
            scores[name] = np.mean(score_list)
        return scores

    def save(self, models_dir):
        """保存。"""
        seeds = [123] if self.seeds is None else self.seeds

        models_dir = pathlib.Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        for fold, gbms_fold in enumerate(self.gbms_):
            for seed, gbm in zip(seeds, gbms_fold):
                gbm.save_model(
                    str(models_dir / f"model.fold{fold}.seed{seed}.txt"),
                    num_iteration=gbm.best_iteration,
                )
        # ついでにfeature_importanceも。
        df_importance = self.feature_importance()
        df_importance.to_excel(str(models_dir / "feature_importance.xlsx"))

    def load(self, models_dir):
        """読み込み。

        Args:
            models_dir (pathlib.Path): 保存先ディレクトリ

        """
        import lightgbm as lgb

        seeds = [123] if self.seeds is None else self.seeds
        self.gbms_ = np.array(
            [
                [
                    lgb.Booster(
                        model_file=str(models_dir / f"model.fold{fold}.seed{seed}.txt")
                    )
                    for seed in seeds
                ]
                for fold in range(self.nfold)
            ]
        )

    def predict(self, dataset):
        """予測結果をリストで返す。

        Args:
            dataset (tk.data.Dataset): 入力データ

        Returns:
            np.ndarray: len(self.folds)個の予測結果

        """
        return np.array(
            [
                np.mean(
                    [
                        gbm.predict(
                            dataset.data[gbm.feature_name()],
                            num_iteration=gbm.best_iteration,
                        )
                        for gbm in gbms_fold
                    ],
                    axis=0,
                )
                for gbms_fold in self.gbms_
            ]
        )

    def feature_importance(self, importance_type="gain"):
        """Feature ImportanceをDataFrameで返す。"""
        import pandas as pd

        columns = self.gbms_[0][0].feature_name()
        for gbms_fold in self.gbms_:
            for gbm in gbms_fold:
                assert tuple(columns) == tuple(gbm.feature_name())

        t = np.int32 if importance_type == "split" else np.float32
        fi = np.zeros((len(columns),), dtype=t)
        for gbms_fold in self.gbms_:
            for gbm in gbms_fold:
                fi += gbm.feature_importance(importance_type=importance_type)

        return pd.DataFrame(data={"importance": fi}, index=columns)


class CBModel(pipeline.Model):
    """CatBoostのモデル。

    Args:
        params (dict): CatBoostのパラメータ
        nfold (int): cvの分割数
        cv_params (dict): catboost.train用kwargs

    """

    def __init__(self, params, nfold, cv_params=None):
        self.params = params
        self.nfold = nfold
        self.cv_params = cv_params
        self.gbms_ = None
        self.train_pool_ = None

    def cv(self, dataset, folds, models_dir):
        """CVして保存。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex
            models_dir (pathlib.Path): 保存先ディレクトリ (Noneなら保存しない)

        Returns:
            dict: metrics名と値

        """
        import catboost

        train_pool = catboost.Pool(
            data=dataset.data,
            label=dataset.labels,
            group_id=dataset.groups,
            feature_names=dataset.data.columns.values.tolist(),
            cat_features=dataset.data.select_dtypes("object").columns.values,
        )

        self.gbms_, score_list = [], []
        for fold, (train_indices, val_indices) in enumerate(folds):
            with tk.log.trace_scope(f"cv({fold + 1}/{len(folds)})"):
                gbm = catboost.train(
                    params=self.params,
                    pool=train_pool.slice(train_indices),
                    eval_set=train_pool.slice(val_indices),
                    **(self.cv_params or {}),
                )
                self.gbms_.append(gbm)
                score_list.append(gbm.get_best_score()["validation"])

        cv_weights = [len(val_indices) for _, val_indices in folds]
        scores = {}
        for k in score_list[0]:
            score = [s[k] for s in score_list]
            score = np.float32(np.average(score, weights=cv_weights))
            scores[k] = score
            tk.log.get(__name__).info(f"cv {k}: {score}")

        if models_dir is not None:
            self.train_pool_ = train_pool
            self.save(models_dir)

        return scores

    def save(self, models_dir):
        """保存。"""
        assert self.train_pool_ is not None
        models_dir = pathlib.Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        for fold, gbm in enumerate(self.gbms_):
            gbm.save_model(
                str(models_dir / f"model.fold{fold}.cbm"), pool=self.train_pool_
            )
        # ついでにfeature_importanceも。
        df_importance = self.feature_importance()
        df_importance.to_excel(str(models_dir / "feature_importance.xlsx"))

    def load(self, models_dir):
        """読み込み。

        Args:
            models_dir (pathlib.Path): 保存先ディレクトリ

        """
        import catboost

        def load(model_path):
            gbm = catboost.CatBoost()
            gbm.load_model(model_path)
            return gbm

        self.gbms_ = [
            load(str(models_dir / f"model.fold{fold}.cbm"))
            for fold in range(self.nfold)
        ]

    def predict(self, dataset):
        """予測結果をリストで返す。

        Args:
            dataset (tk.data.Dataset): 入力データ

        Returns:
            np.ndarray: len(self.folds)個の予測結果

        """
        return np.array([gbm.predict(dataset.data) for gbm in self.gbms_])

    def feature_importance(self):
        """Feature ImportanceをDataFrameで返す。"""
        import pandas as pd

        columns = self.gbms_[0].feature_names_
        for gbm in self.gbms_:
            assert tuple(columns) == tuple(gbm.feature_names_)

        fi = np.zeros((len(columns),), dtype=np.float32)
        for gbm in self.gbms_:
            fi += gbm.get_feature_importance()

        return pd.DataFrame(data={"importance": fi}, index=columns)


class XGBModel(pipeline.Model):
    """XGBoostのモデル。

    Args:
        params (dict): XGBoostのパラメータ
        nfold (int): cvの分割数
        early_stopping_rounds (int): xgb.cvのパラメータ
        num_boost_round (int): xgb.cvのパラメータ
        verbose_eval (int): xgb.cvのパラメータ
        callbacks (array-like): xgb.cvのパラメータ
        cv_params (dict): xgb.cvのパラメータ (**kwargs)
        seeds (array-like): seed ensemble用のseedの配列

    """

    def __init__(
        self,
        params,
        nfold,
        early_stopping_rounds=200,
        num_boost_round=9999,
        verbose_eval=100,
        callbacks=None,
        cv_params=None,
    ):
        self.params = params
        self.nfold = nfold
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.verbose_eval = verbose_eval
        self.callbacks = callbacks
        self.cv_params = cv_params
        self.gbms_ = None

    def cv(self, dataset, folds, models_dir):
        """CVして保存。

        Args:
            dataset (tk.data.Dataset): 入力データ
            folds (list): CVのindex
            models_dir (pathlib.Path): 保存先ディレクトリ (Noneなら保存しない)

        Returns:
            dict: metrics名と値

        """
        import xgboost as xgb

        train_set = xgb.DMatrix(
            data=dataset.data,
            label=dataset.labels,
            weight=dataset.weights,
            feature_names=dataset.data.columns.values,
        )

        self.gbms_ = []

        def model_extractor(env):
            self.gbms_.clear()
            self.gbms_.extend([f.bst for f in env.cvfolds])

        eval_hist = xgb.cv(
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

        if models_dir is not None:
            self.save(models_dir)

        return scores

    def save(self, models_dir):
        """保存。"""
        models_dir = pathlib.Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        for fold, gbm in enumerate(self.gbms_):
            tk.utils.dump(gbm, models_dir / f"model.fold{fold}.pkl")
        # ついでにfeature_importanceも。
        df_importance = self.feature_importance()
        df_importance.to_excel(str(models_dir / "feature_importance.xlsx"))

    def load(self, models_dir):
        """読み込み。

        Args:
            models_dir (pathlib.Path): 保存先ディレクトリ

        """
        self.gbms_ = [
            tk.utils.load(models_dir / f"model.fold{fold}.pkl")
            for fold in range(self.nfold)
        ]

    def predict(self, dataset):
        """予測結果をリストで返す。

        Args:
            dataset (tk.data.Dataset): 入力データ

        Returns:
            np.ndarray: len(self.folds)個の予測結果

        """
        import xgboost as xgb

        data = xgb.DMatrix(data=dataset.data, feature_names=dataset.data.columns.values)
        return np.array([gbm.predict(data) for gbm in self.gbms_])

    def feature_importance(self, importance_type="gain"):
        """Feature ImportanceをDataFrameで返す。"""
        import pandas as pd

        columns = self.gbms_[0].feature_names
        for gbm in self.gbms_:
            assert tuple(columns) == tuple(gbm.feature_names)

        fi = np.zeros((len(columns),), dtype=np.float32)
        for gbm in self.gbms_:
            d = gbm.get_score(importance_type=importance_type)
            fi += [d.get(c, 0) for c in columns]

        return pd.DataFrame(data={"importance": fi}, index=columns)


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
