"""LightGBM"""
from __future__ import annotations

import logging
import pathlib
import typing

import numpy as np
import pandas as pd
import sklearn.metrics

import pytoolkit as tk

from .core import Model

logger = logging.getLogger(__name__)


class LGBModel(Model):
    """LightGBMのモデル。

    Args:
        params: lgb.cvのパラメータ
        nfold: cvの分割数
        models_dir: 保存先ディレクトリ
        early_stopping_rounds: lgb.cvのパラメータ
        num_boost_round: lgb.cvのパラメータ
        verbose_eval: lgb.cvのパラメータ
        callbacks: lgb.cvのパラメータ
        cv_params: lgb.cvのパラメータ (`**kwargs`)
        seeds: seed ensemble用のseedの配列
        init_score: trainとtestのinit_score

    """

    def __init__(
        self,
        params: dict[str, typing.Any],
        nfold: int,
        models_dir: tk.typing.PathLike,
        early_stopping_rounds: int = 200,
        num_boost_round: int = 9999,
        verbose_eval: int = 100,
        callbacks: typing.Sequence[typing.Callable[[typing.Any], None]] = None,
        cv_params: dict[str, typing.Any] = None,
        seeds: np.ndarray | None = None,
        init_score: np.ndarray | None = None,
        preprocessors: tk.pipeline.EstimatorListType = None,
        postprocessors: tk.pipeline.EstimatorListType = None,
    ):
        super().__init__(nfold, models_dir, preprocessors, postprocessors)
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.verbose_eval = verbose_eval
        self.callbacks = callbacks
        self.cv_params = cv_params
        self.seeds = seeds
        self.init_score = init_score
        self.gbms_: np.ndarray | None = None

    def _save(self, models_dir: pathlib.Path):
        assert self.gbms_ is not None
        seeds = np.array([123]) if self.seeds is None else self.seeds

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

    def _load(self, models_dir: pathlib.Path):
        import lightgbm as lgb

        seeds = np.array([123]) if self.seeds is None else self.seeds
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

    def _cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType) -> None:
        import lightgbm as lgb

        if isinstance(dataset.data, pd.DataFrame):
            num_features = len(dataset.data.columns)
        else:
            assert isinstance(dataset.data, np.ndarray)
            assert dataset.data.ndim == 2
            num_features = dataset.data.shape[1]
        assert isinstance(dataset.labels, np.ndarray)

        params = self.params.copy()
        weight = dataset.weights if dataset.weights is not None else None
        # 独自拡張: sklearn風の指定
        if params.get("feature_fraction") == "sqrt":
            params["feature_fraction"] = np.sqrt(num_features) / num_features
            logger.info(f'feature_fraction: "sqrt" -> {params["feature_fraction"]:.3f}')
        elif params.get("feature_fraction") == "log2":
            params["feature_fraction"] = np.log2(num_features) / num_features
            logger.info(f'feature_fraction: "log2" -> {params["feature_fraction"]:.3f}')
        # 独自拡張: クラス別の重みの指定
        if "class_weights" in params:
            if params.get("class_weights") == "balanced":
                assert "num_class" in params, str(params)
                class_weights = params["num_class"] / np.bincount(dataset.labels)
            else:
                class_weights = params.get("class_weights")
            params.pop("class_weights")
            sample_weights = class_weights[dataset.labels]
            if weight is None:
                weight = sample_weights
            else:
                weight *= sample_weights

        train_set = lgb.Dataset(
            dataset.data,
            dataset.labels,
            weight=weight,
            group=np.bincount(dataset.groups) if dataset.groups is not None else None,
            init_score=dataset.init_score if dataset.init_score is not None else None,
            free_raw_data=False,
        )

        seeds = np.array([123]) if self.seeds is None else self.seeds

        scores_list: dict[str, list[float]] = {}
        self.gbms_ = np.empty((len(folds), len(seeds)), dtype=object)
        for seed_i, seed in enumerate(seeds):
            with tk.log.trace(f"seed averaging({seed_i + 1}/{len(seeds)})"):
                model_extractor = ModelExtractor()
                eval_hist = lgb.cv(
                    params,
                    train_set,
                    folds=folds,
                    early_stopping_rounds=self.early_stopping_rounds,
                    num_boost_round=self.num_boost_round,
                    callbacks=list(self.callbacks or [])
                    + [model_extractor, EvaluationLogger(period=self.verbose_eval)],
                    seed=seed,
                    **(self.cv_params or {}),
                )
                logger.info(f"best iteration: {model_extractor.best_iteration}")
                for k in eval_hist:
                    if k.endswith("-mean"):
                        name, score = k[:-5], float(eval_hist[k][-1])
                        if name not in scores_list:
                            scores_list[name] = []
                        scores_list[name].append(score)
                        logger.info(f"cv {name}: {score}")
                self.gbms_[:, seed_i] = model_extractor.raw_boosters
                # 怪しいけどとりあえずいったん書き換えちゃう
                for gbm in self.gbms_[:, seed_i]:
                    gbm.best_iteration = model_extractor.best_iteration

        scores = {
            name: np.mean(score_list).tolist()
            for name, score_list in scores_list.items()
        }
        for k, v in scores.items():
            logger.info(f"cv(mean) {k}: {v:,.3f}")

    def _predict(
        self, dataset: tk.data.Dataset, fold: int
    ) -> np.ndarray | list[np.ndarray]:
        assert self.gbms_ is not None

        def _get_data(gbm):
            if isinstance(dataset.data, pd.DataFrame):
                return dataset.data[gbm.feature_name()]
            return dataset.data

        pred = np.mean(
            [
                gbm.predict(_get_data(gbm), num_iteration=gbm.best_iteration)
                for gbm in self.gbms_[fold, :]
            ],
            axis=0,
        )
        if dataset.init_score is not None:
            pred += dataset.init_score
        return pred

    def feature_importance(self, importance_type: str = "gain"):
        """Feature ImportanceをDataFrameで返す。"""
        import lightgbm as lgb

        assert self.gbms_ is not None
        columns = self.gbms_[0][0].feature_name()
        for gbms_fold in self.gbms_:
            for gbm in gbms_fold:
                assert tuple(columns) == tuple(gbm.feature_name())

        t = np.int32 if importance_type == "split" else np.float32
        fi: typing.Any = np.zeros((len(columns),), dtype=t)
        for gbms_fold in self.gbms_:
            for gbm in typing.cast(typing.Iterable[lgb.Booster], gbms_fold):
                fi += gbm.feature_importance(importance_type=importance_type)

        return pd.DataFrame(data={"importance": fi}, index=columns)


class ModelExtractor:
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
        if (
            self.period > 0
            and env.evaluation_result_list
            and (env.iteration + 1) % self.period == 0
        ):
            from lightgbm.callback import _format_eval_result

            result = "\t".join(
                [
                    _format_eval_result(x, self.show_stdv)
                    for x in env.evaluation_result_list
                ]
            )
            logger.log(self.level, f"[{env.iteration + 1}]\t{result}")


def lgb_r2(preds, train_data):
    """LightGBM用R2"""
    labels = train_data.get_label()
    return "r2", sklearn.metrics.r2_score(labels, preds), True
