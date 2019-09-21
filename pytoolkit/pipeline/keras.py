"""Keras"""
from __future__ import annotations

import pathlib
import typing

import numpy as np
import tensorflow as tf

import pytoolkit as tk

from .. import keras
from .core import Model


class KerasModel(Model):
    """Kerasのモデル。

    _saveではなく学習時にsaveしてしまっているので注意。

    Args:
        create_model_fn: モデルを作成する関数。
        train_data_loader: 訓練データの読み込み
        val_data_loader: 検証データの読み込み
        models_dir (PathLike): 保存先ディレクトリ
        model_name_format: モデルのファイル名のフォーマット。{fold}のところに数字が入る。
        skip_if_exists: モデルが存在してもスキップせず再学習するならFalse。
        fit_params: tk.models.fit()のパラメータ
        load_model_fn: モデルを読み込む関数
        use_horovod: MPIによる分散処理をするか否か。
        parallel_cv: lgb.cvなどのように全foldまとめて処理するならTrue

    """

    def __init__(
        self,
        create_model_fn: typing.Callable[[], tk.keras.models.Model],
        train_data_loader: tk.data.DataLoader,
        val_data_loader: tk.data.DataLoader,
        *,
        models_dir,
        model_name_format: str = "model.fold{fold}.h5",
        skip_if_exists: bool = True,
        fit_params: dict = None,
        load_model_fn: typing.Callable[[pathlib.Path], tk.keras.models.Model] = None,
        use_horovod: bool = False,
        parallel_cv: bool = False,
        data_loaders: list = None,
        postprocessors: list = None,
    ):
        super().__init__(data_loaders, postprocessors)
        self.create_model_fn = create_model_fn
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.models_dir = pathlib.Path(models_dir)
        self.model_name_format = model_name_format
        self.skip_if_exists = skip_if_exists
        self.fit_params = fit_params
        self.load_model_fn = load_model_fn
        self.use_horovod = use_horovod
        self.parallel_cv = parallel_cv
        self.models: typing.Optional[typing.List[tk.keras.models.Model]] = None

    def _save(self, models_dir):
        assert models_dir == self.models_dir

    def _load(self, models_dir):
        assert models_dir == self.models_dir
        load_model_fn = self.load_model_fn or tk.models.load
        self.models = []
        for fold in range(999):
            model_path = models_dir / self.model_name_format.format(fold=fold)
            if model_path.exists():
                self.models.append(load_model_fn(model_path))
            else:
                break

    def _cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType) -> dict:
        if self.parallel_cv:
            return self._parallel_cv(dataset, folds)
        else:
            return self._serial_cv(dataset, folds)

    def _parallel_cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType):
        self.models = [self.create_model_fn() for _ in folds]

        inputs = []
        targets = []
        outputs = []
        losses = []
        metrics: dict = {n: [] for n in self.models[0].metrics_names if n != "loss"}
        for i, model in enumerate(self.models):
            input_shape = model.input_shape
            output_shape = model.output_shape
            if isinstance(input_shape, tuple):
                input_shape = [input_shape]
            if isinstance(output_shape, tuple):
                output_shape = [output_shape]

            model_inputs = [
                keras.layers.Input(s[1:], name=f"model{i}_input{j}")
                for j, s in enumerate(input_shape)
            ]
            model_targets = [
                keras.layers.Input(s[1:], name=f"model{i}_target{j}")
                for j, s in enumerate(output_shape)
            ]
            inputs.extend(model_inputs)
            targets.extend(model_targets)
            if len(model_targets) == 1:
                model_targets = model_targets[0]

            x = model(model_inputs)
            outputs.append(x)
            losses.extend([loss(model_targets, x) for loss in model.loss_functions])
            assert len(metrics) == len(model.metrics)
            for k, m in zip(metrics, model.metrics):
                metrics[k].append(m(model_targets, x))

        def loss(y_true, y_pred):
            del y_true, y_pred
            return tf.reduce_mean(losses, axis=0)

        for k, v in metrics.items():

            def metric_func(y_true, y_pred, v=v):
                del y_true, y_pred
                return tf.reduce_mean(v, axis=0)

            metric_func.__name__ = k
            metrics[k] = metric_func

        model = keras.models.Model(inputs=inputs + targets, outputs=outputs)
        model.compile(self.models[0].optimizer, loss, [m for m in metrics.values()])
        tk.models.summary(model)

        def generator(datasets, data_loader):
            iterators = [
                data_loader.iter(dataset, shuffle=True, use_horovod=True).run()
                for dataset in datasets
            ]
            while True:
                X_batch = {}
                for i, it in enumerate(iterators):
                    Xt, yt = next(it, (None, None))
                    assert Xt is not None
                    assert yt is not None

                    if isinstance(Xt, np.ndarray):
                        Xt = [Xt]
                    elif isinstance(Xt, dict):
                        Xt = Xt.values()  # TODO: 並び順
                    for j, Xtj in enumerate(Xt):
                        X_batch[f"model{i}_input{j}"] = Xtj

                    if isinstance(yt, np.ndarray):
                        yt = [yt]
                    elif isinstance(yt, dict):
                        yt = yt.values()  # TODO: 並び順
                    for j, ytj in enumerate(yt):
                        X_batch[f"model{i}_target{j}"] = ytj
                yield X_batch, None

        train_sets = []
        val_sets = []
        for train_indices, val_indices in folds:
            train_sets.append(dataset.slice(train_indices))
            val_sets.append(dataset.slice(val_indices))

        model.fit_generator(
            generator(train_sets, self.train_data_loader),
            steps_per_epoch=-(-len(train_sets[0]) // self.train_data_loader.batch_size),
            validation_data=generator(val_sets, self.val_data_loader),
            validation_steps=-(-len(val_sets[0]) // self.val_data_loader.batch_size),
            **(self.fit_params or {}),
        )

        evals = model.evaluate_generator(
            generator(val_sets, self.val_data_loader),
            -(-len(val_sets[0]) // self.val_data_loader.batch_size) * 3,
        )
        scores = dict(zip(model.metrics_names, evals))
        for k, v in scores.items():
            tk.log.get(__name__).info(f"CV: val_{k}={v:,.3f}")

        return scores

    def _serial_cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType):
        self.models = []
        score_list = []
        score_weights = []
        for fold, (train_indices, val_indices) in enumerate(folds):
            tk.log.get(__name__).info(
                f"Fold {fold + 1}/{len(folds)}: train={len(train_indices)} val={len(val_indices)}"
            )
            train_set = dataset.slice(train_indices)
            val_set = dataset.slice(val_indices)
            model = self.create_model_fn()
            tk.models.fit(
                model,
                train_set=train_set,
                train_data_loader=self.train_data_loader,
                val_set=val_set,
                val_data_loader=self.val_data_loader,
                **(self.fit_params or {}),
            )
            model_path = self.models_dir / self.model_name_format.format(fold=fold)
            tk.models.save(model, model_path)
            self.models.append(model)
            scores = tk.models.evaluate(
                model, val_set, data_loader=self.val_data_loader, use_horovod=True
            )
            for k, v in scores.items():
                tk.log.get(__name__).info(
                    f"Fold {fold + 1}/{len(folds)}: val_{k}={v:,.3f}"
                )
            score_list.append(scores)
            score_weights.append(len(val_indices))

        return {
            k: np.average([s[k] for s in score_list], weights=score_weights)
            for k in score_list[0]
        }

    def _predict(self, dataset: tk.data.Dataset) -> typing.List[np.ndarray]:
        assert self.models is not None
        return [
            tk.models.predict(
                model,
                dataset,
                self.val_data_loader,
                desc=f"fold",
                use_horovod=self.use_horovod,
                # TODO: TTA
            )
            for model in self.models
        ]
