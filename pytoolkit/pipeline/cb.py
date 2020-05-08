"""CatBoost"""
from __future__ import annotations

import pathlib
import typing

import numpy as np
import pandas as pd

import pytoolkit as tk

from .core import Model


class CBModel(Model):
    """CatBoostのモデル。

    Args:
        params: CatBoostのパラメータ
        nfold: cvの分割数
        models_dir: 保存先ディレクトリ
        cv_params: catboost.train用パラメータ (`**kwargs`)

    """

    def __init__(
        self,
        params: typing.Dict[str, typing.Any],
        nfold: int,
        models_dir: tk.typing.PathLike,
        cv_params: typing.Dict[str, typing.Any] = None,
        preprocessors=None,
        postprocessors=None,
    ):
        import catboost

        super().__init__(nfold, models_dir, preprocessors, postprocessors)
        self.params = params
        self.cv_params = cv_params
        self.gbms_: typing.Optional[typing.List[catboost.CatBoost]] = None
        self.train_pool_: catboost.Pool = None

    def _save(self, models_dir: pathlib.Path):
        assert self.gbms_ is not None
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

    def _load(self, models_dir: pathlib.Path):
        import catboost

        def load(model_path):
            gbm = catboost.CatBoost()
            gbm.load_model(model_path)
            return gbm

        self.gbms_ = [
            load(str(models_dir / f"model.fold{fold}.cbm"))
            for fold in range(self.nfold)
        ]

    def _cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType) -> None:
        import catboost

        assert isinstance(dataset.data, pd.DataFrame)

        self.train_pool_ = catboost.Pool(
            data=dataset.data,
            label=dataset.labels,
            group_id=dataset.groups,
            feature_names=dataset.data.columns.values.tolist(),
            cat_features=dataset.data.select_dtypes("object").columns.values,
        )

        self.gbms_, score_list = [], []
        for fold, (train_indices, val_indices) in enumerate(folds):
            with tk.log.trace(f"fold{fold}"):
                gbm = catboost.train(
                    params=self.params,
                    pool=self.train_pool_.slice(train_indices),
                    eval_set=self.train_pool_.slice(val_indices),
                    **(self.cv_params or {}),
                )
                self.gbms_.append(gbm)
                score_list.append(gbm.get_best_score()["validation"])

        cv_weights = [len(val_indices) for _, val_indices in folds]
        evals: tk.evaluations.EvalsType = {}
        for k in score_list[0]:
            score = [s[k] for s in score_list]
            score = np.float32(np.average(score, weights=cv_weights))
            evals[k] = score
        tk.log.get(__name__).info(f"cv: {tk.evaluations.to_str(evals)}")

    def _predict(self, dataset: tk.data.Dataset, fold: int) -> np.ndarray:
        assert self.gbms_ is not None
        if self.params.get("loss_function") in ("MultiClass",):  # TODO
            prediction_type = "Probability"
        else:
            prediction_type = "RawFormulaVal"
        return self.gbms_[fold].predict(dataset.data, prediction_type=prediction_type)

    def feature_importance(self):
        """Feature ImportanceをDataFrameで返す。"""
        assert self.gbms_ is not None
        columns = self.gbms_[0].feature_names_
        for gbm in self.gbms_:
            assert tuple(columns) == tuple(gbm.feature_names_)

        fi = np.zeros((len(columns),), dtype=np.float32)
        for gbm in self.gbms_:
            fi += gbm.get_feature_importance()

        return pd.DataFrame(data={"importance": fi}, index=columns)
