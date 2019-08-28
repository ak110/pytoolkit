import pathlib

import numpy as np
import tensorflow as tf

import pytoolkit as tk

from .. import keras
from ._core import Model


class KerasModel(Model):
    """Kerasのモデル。

    _saveではなく学習時にsaveしてしまっているので注意。

    Args:
        create_model_fn (callable): モデルを作成する関数。
        train_preprocessor (tk.data.Preprocessor): 訓練データの前処理
        val_preprocessor (tk.data.Preprocessor): 検証データの前処理
        batch_size (int): バッチサイズ
        models_dir (PathLike): 保存先ディレクトリ
        model_name_format (str): モデルのファイル名のフォーマット。{fold}のところに数字が入る。
        skip_if_exists (bool): モデルが存在してもスキップせず再学習するならFalse。
        fit_params (dict): tk.models.fit()のパラメータ
        load_model_fn (callable): モデルを読み込む関数
        use_horovod (bool): MPIによる分散処理をするか否か。
        parallel_cv (bool): lgb.cvなどのように全foldまとめて処理するならTrue

    """

    def __init__(
        self,
        create_model_fn,
        train_preprocessor,
        val_preprocessor,
        batch_size=32,
        *,
        models_dir,
        model_name_format="model.fold{fold}.h5",
        skip_if_exists=True,
        fit_params=None,
        load_model_fn=None,
        use_horovod=False,
        parallel_cv=False,
        preprocessors=None,
        postprocessors=None,
    ):
        super().__init__(preprocessors, postprocessors)
        self.create_model_fn = create_model_fn
        self.train_preprocessor = train_preprocessor
        self.val_preprocessor = val_preprocessor
        self.batch_size = batch_size
        self.models_dir = pathlib.Path(models_dir)
        self.model_name_format = model_name_format
        self.skip_if_exists = skip_if_exists
        self.fit_params = fit_params
        self.load_model_fn = load_model_fn
        self.use_horovod = use_horovod
        self.parallel_cv = parallel_cv
        self.models = None

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

    def _cv(self, dataset, folds):
        if self.parallel_cv:
            return self._parallel_cv(dataset, folds)
        else:
            return self._serial_cv(dataset, folds)

    def _parallel_cv(self, dataset, folds):
        self.models = [self.create_model_fn() for _ in folds]

        inputs = []
        targets = []
        outputs = []
        losses = []
        metrics = {n: [] for n in self.models[0].metrics_names if n != "loss"}
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

        def generator(datasets, preprocessor):
            iterators = [
                tk.data.DataLoader(
                    dataset,
                    preprocessor,
                    self.batch_size,
                    shuffle=True,
                    parallel=True,
                    use_horovod=True,
                ).run()
                for dataset in datasets
            ]
            while True:
                X_batch = {}
                for i, it in enumerate(iterators):
                    Xt, yt = next(it, None)

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

        train_datasets = []
        val_datasets = []
        for train_indices, val_indices in folds:
            train_datasets.append(dataset.slice(train_indices))
            val_datasets.append(dataset.slice(val_indices))

        model.fit_generator(
            generator(train_datasets, self.train_preprocessor),
            steps_per_epoch=-(-len(train_datasets[0]) // self.batch_size),
            validation_data=generator(val_datasets, self.val_preprocessor),
            validation_steps=-(-len(val_datasets[0]) // self.batch_size),
            **(self.fit_params or {}),
        )

        evals = model.evaluate_generator(
            generator(val_datasets, self.val_preprocessor),
            -(-len(val_datasets[0]) // self.batch_size) * 3,
        )
        scores = dict(zip(model.metrics_names, evals))
        for k, v in scores.items():
            tk.log.get(__name__).info(f"CV: val_{k}={v:,.3f}")

        return scores

    def _serial_cv(self, dataset, folds):
        self.models = []
        score_list = []
        score_weights = []
        for fold, (train_indices, val_indices) in enumerate(folds):
            tk.log.get(__name__).info(
                f"Fold {fold + 1}/{len(folds)}: train={len(train_indices)} val={len(val_indices)}"
            )
            train_dataset = dataset.slice(train_indices)
            val_dataset = dataset.slice(val_indices)
            model = self.create_model_fn()
            tk.models.fit(
                model,
                train_dataset=train_dataset,
                train_preprocessor=self.train_preprocessor,
                val_dataset=val_dataset,
                val_preprocessor=self.val_preprocessor,
                batch_size=self.batch_size,
                **(self.fit_params or {}),
            )
            model_path = self.models_dir / self.model_name_format.format(fold=fold)
            tk.models.save(model, model_path)
            self.models.append(model)
            scores = tk.models.evaluate(
                model,
                val_dataset,
                preprocessor=self.val_preprocessor,
                batch_size=self.batch_size,
                use_horovod=True,
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

    def _predict(self, dataset):
        return [
            tk.models.predict(
                model,
                dataset,
                self.val_preprocessor,
                self.batch_size,
                desc=f"fold",
                use_horovod=self.use_horovod,
                # TODO: TTA
            )
            for model in self.models
        ]
