"""Keras"""
from __future__ import annotations

import abc
import pathlib
import typing

import numpy as np
import tensorflow as tf

import pytoolkit as tk

from .core import Model


class KerasModel(Model, metaclass=abc.ABCMeta):
    """Kerasのモデル。

    _saveではなく学習時にsaveしてしまっているので注意。

    Args:
        train_data_loader: 訓練データの読み込み
        val_data_loader: 検証データの読み込み
        refine_data_loader: Refined Data Augmentation <https://arxiv.org/abs/1909.09148> 用
        models_dir: 保存先ディレクトリ
        model_name_format: モデルのファイル名のフォーマット。{fold}のところに数字が入る。
        skip_if_exists: cv()でモデルが存在するときスキップするならTrue。
        skip_folds: cv()で（skip_if_existsとは関係なく）スキップするfold。[0, nfold)
        epochs: tk.models.fit()のパラメータ
        refine_epochs: tk.models.fit()のパラメータ (refine時)
        callbacks: tk.models.fit()のパラメータ
        fit_params: tk.models.fit()のパラメータ
        use_horovod: 推論時にMPIによる分散処理をするか否か。(学習時は常にTrue)
        parallel_cv: lgb.cvなどのように全foldまとめて処理するならTrue

    model_name_formatに"{fold}"が含まれない名前を指定した場合、
    cvではなくtrainを使うモードということにする。

    事前にself.modelsにモデルがある状態でcvやfitを呼んだ場合は追加学習ということにする。

    """

    def __init__(
        self,
        train_data_loader: tk.data.DataLoader,
        val_data_loader: tk.data.DataLoader,
        refine_data_loader: tk.data.DataLoader = None,
        *,
        epochs: int,
        refine_epochs: int = 50,
        callbacks: typing.List[tf.keras.callbacks.Callback] = None,
        models_dir: tk.typing.PathLike,
        model_name_format: str = "model.fold{fold}.h5",
        skip_if_exists: bool = True,
        skip_folds: typing.Sequence[int] = (),
        fit_params: dict = None,
        use_horovod: bool = False,
        parallel_cv: bool = False,
        on_batch_fn: tk.models.OnBatchFnType = None,
        preprocessors: tk.pipeline.EstimatorListType = None,
        postprocessors: tk.pipeline.EstimatorListType = None,
    ):
        super().__init__(preprocessors, postprocessors)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.refine_data_loader = refine_data_loader
        self.models_dir = pathlib.Path(models_dir)
        self.model_name_format = model_name_format
        self.skip_if_exists = skip_if_exists
        self.skip_folds = skip_folds
        self.epochs = epochs
        self.refine_epochs = refine_epochs
        self.callbacks = callbacks
        self.fit_params = fit_params
        self.use_horovod = use_horovod
        self.parallel_cv = parallel_cv
        self.on_batch_fn = on_batch_fn
        self.models: typing.Optional[typing.List[tf.keras.models.Model]] = None
        if self.parallel_cv:
            assert self.refine_data_loader is None, "NotImplemented"

    def _save(self, models_dir: pathlib.Path):
        assert models_dir == self.models_dir

    def _load(self, models_dir: pathlib.Path):
        self.models = []
        for fold in range(999):
            model_path = models_dir / self.model_name_format.format(fold=fold + 1)
            if model_path.exists():
                self.models.append(self.load_model(model_path))
                if "{fold}" not in self.model_name_format:
                    break
            else:
                break

    def _cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType) -> dict:
        assert "{fold}" in self.model_name_format
        if self.parallel_cv:
            return self._parallel_cv(dataset, folds)
        else:
            return self._serial_cv(dataset, folds)

    def _parallel_cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType):
        self.models = [self.create_model() for _ in folds]

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
                tf.keras.layers.Input(s[1:], name=f"model{i}_input{j}")
                for j, s in enumerate(input_shape)
            ]
            model_targets = [
                tf.keras.layers.Input(s[1:], name=f"model{i}_target{j}")
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

        model = tf.keras.models.Model(inputs=inputs + targets, outputs=outputs)
        model.compile(self.models[0].optimizer, loss, list(metrics.values()))
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

        model.fit(
            generator(train_sets, self.train_data_loader),
            steps_per_epoch=-(-len(train_sets[0]) // self.train_data_loader.batch_size),
            validation_data=generator(val_sets, self.val_data_loader),
            validation_steps=-(-len(val_sets[0]) // self.val_data_loader.batch_size),
            epochs=self.epochs,
            callbacks=self.callbacks,
            **(self.fit_params or {}),
        )

        evals = model.evaluate(
            generator(val_sets, self.val_data_loader),
            -(-len(val_sets[0]) // self.val_data_loader.batch_size) * 3,
        )
        scores = dict(zip(model.metrics_names, evals))
        for k, v in scores.items():
            tk.log.get(__name__).info(f"CV: val_{k}={v:,.3f}")

        return scores

    def _serial_cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType):
        score_list = []
        score_weights = []
        for fold, (train_indices, val_indices) in enumerate(folds):
            tk.log.get(__name__).info(
                f"Fold {fold + 1}/{len(folds)}: train={len(train_indices)} val={len(val_indices)}"
            )
            train_set = dataset.slice(train_indices)
            val_set = dataset.slice(val_indices)
            scores = self.train(train_set, val_set, fold=fold)
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
                use_horovod=self.use_horovod,
                on_batch_fn=self.on_batch_fn,
            )
            for model in self.models
        ]

    def check(self) -> KerasModel:
        """モデルの動作確認。(KerasModel独自メソッド)

        Returns:
            self

        """
        model = self.create_model()
        # summary表示
        tk.models.summary(model)
        # グラフを出力
        try:
            tk.models.plot(model, self.models_dir / "model.svg")
        except ValueError:
            pass  # "Cannot embed the 'svg' image format" (tf >= 1.14)
        return self

    @typing.overload
    def train(
        self, train_set: tk.data.Dataset, val_set: None = None, fold: int = 0
    ) -> None:
        pass

    @typing.overload
    def train(
        self, train_set: tk.data.Dataset, val_set: tk.data.Dataset, fold: int = 0
    ) -> typing.Dict[str, float]:
        # pylint: disable=function-redefined
        pass

    def train(
        self, train_set: tk.data.Dataset, val_set: tk.data.Dataset = None, fold: int = 0
    ) -> typing.Optional[typing.Dict[str, float]]:
        """1fold分の学習。(KerasModel独自メソッド)

        Args:
            train_set: 訓練データ
            val_set: 検証データ
            fold: 何番目のモデルを使うか

        Returns:
            メトリクス名と値のdict

        """
        # pylint: disable=function-redefined
        tk.hvd.barrier()
        tk.log.get(__name__).info(
            f"train: {len(train_set)} samples, val: {len(val_set) if val_set is not None else 0} samples, batch_size: {self.train_data_loader.batch_size}x{tk.hvd.size()}"
        )

        if self.models is None:
            self.models = []
        while len(self.models) <= fold:
            self.models.append(None)

        model_path = self.models_dir / self.model_name_format.format(fold=fold + 1)

        if self.skip_if_exists and model_path.exists():
            tk.log.get(__name__).info(
                f"Fold {fold + 1}: Loading '{model_path}'... (skip_if_exists)"
            )
            self.models[fold] = self.load_model(model_path)
            trained = False
        elif fold in self.skip_folds and model_path.exists():
            tk.log.get(__name__).info(
                f"Fold {fold + 1}: Loading '{model_path}'... (skip_folds)"
            )
            self.models[fold] = self.load_model(model_path)
            trained = False
        else:
            if self.models[fold] is not None:
                model = self.models[fold]  # 既にあればそれを使う
            else:
                model = self.create_model()
            trained = True

            # fit
            tk.hvd.barrier()
            tk.models.fit(
                model,
                train_set=train_set,
                train_data_loader=self.train_data_loader,
                val_set=val_set,
                val_data_loader=self.val_data_loader,
                epochs=self.epochs,
                callbacks=self.callbacks,
                **(self.fit_params or {}),
            )

            # refine
            if self.refine_data_loader is not None:
                tk.log.get(__name__).info(
                    f"Fold {fold + 1}: Refining {self.refine_epochs} epochs..."
                )
                tk.models.freeze_layers(model, tf.keras.layers.BatchNormalization)
                self.compile_model(model, mode="refine")
                tk.models.fit(
                    model,
                    train_set=train_set,
                    train_data_loader=self.refine_data_loader,
                    val_set=val_set,
                    val_data_loader=self.val_data_loader,
                    epochs=self.refine_epochs,
                )

            tk.models.save(model, model_path)
            self.models[fold] = model

        # 訓練データと検証データの評価
        tk.hvd.barrier()
        self.evaluate(train_set, prefix="", fold=fold)
        if val_set is None:
            return None
        evals = self.evaluate(val_set, prefix="val_", fold=fold)

        # メモリを食いがちなので学習完了後は再構築する
        if trained:
            del model
            self.models[fold] = None
            tk.hvd.clear()
            self.models[fold] = self.load_model(model_path)

        return evals

    def evaluate(
        self, dataset: tk.data.Dataset, prefix: str = "", fold: int = 0
    ) -> typing.Dict[str, float]:
        """評価して結果をINFOログ出力する。(KerasModel独自メソッド)

        Args:
            dataset: データ
            prefix: メトリクス名の接頭文字列
            fold: 何番目のモデルを使うか

        Returns:
            metricsの文字列と値のdict

        """
        assert self.models is not None
        assert self.preprocessors is None  # とりあえず未対応
        assert self.postprocessors is None  # とりあえず未対応
        evals = tk.models.evaluate(
            self.models[fold],
            dataset,
            data_loader=self.val_data_loader,  # DataAugmentation無しで評価
            prefix=prefix,
            use_horovod=self.use_horovod,
        )
        if tk.hvd.is_master():
            max_len = max(len(n) for n in evals)
            for n, v in evals.items():
                tk.log.get(__name__).info(f'{n}:{" " * (max_len - len(n))} {v:.3f}')
        tk.hvd.barrier()
        return evals

    def predict_flow(
        self, dataset: tk.data.Dataset, fold: int = 0
    ) -> typing.Iterator[tk.models.ModelIOType]:
        """予測。

        Args:
            dataset: データセット
            fold: 何番目のモデルを使うか

        """
        assert self.models is not None
        assert self.preprocessors is None  # とりあえず未対応
        assert self.postprocessors is None  # とりあえず未対応
        return tk.models.predict_flow(
            self.models[fold],
            dataset,
            self.val_data_loader,
            on_batch_fn=self.on_batch_fn,
        )

    def load_model(self, model_path: pathlib.Path) -> tf.keras.models.Model:
        """モデルの読み込み。必要に応じてオーバーライドする。"""
        model = self.create_model()
        tk.models.load_weights(model, model_path)
        return model

    def create_model(self, mode: str = "train") -> tf.keras.models.Model:
        """モデルの作成。必要に応じてオーバーライドする。"""
        model = self.create_network()
        self.compile_model(model, mode)
        return model

    def compile_model(self, model: tf.keras.models.Mode, mode: str = "train") -> None:
        """モデルのコンパイル。必要に応じてオーバーライドする。"""
        optimizer = self.create_optimizer(mode)
        loss, metrics = self.create_loss(model)
        tk.models.compile(model, optimizer, loss, metrics)

    @abc.abstractmethod
    def create_network(self) -> tf.keras.models.Model:
        """ネットワークの作成。"""

    @abc.abstractmethod
    def create_optimizer(self, mode: str) -> tk.models.OptimizerType:
        """optimizerの作成。"""

    @abc.abstractmethod
    def create_loss(self, model: tf.keras.models.Model) -> tuple:
        """lossとmetricsの作成。"""
