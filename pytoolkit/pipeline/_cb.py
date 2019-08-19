import pathlib

import numpy as np

import pytoolkit as tk

from ._core import Model


class CBModel(Model):
    """CatBoostのモデル。

    Args:
        params (dict): CatBoostのパラメータ
        nfold (int): cvの分割数
        cv_params (dict): catboost.train用パラメータ (`**kwargs`)

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
