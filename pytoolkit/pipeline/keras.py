"""Keras"""
from __future__ import annotations

import gc
import pathlib
import typing

import numpy as np
import tensorflow as tf

import pytoolkit as tk

from .core import Model


class KerasModel(Model):
    """Kerasのモデル。

    _saveではなく学習時にsaveしてしまっているので注意。

    Args:
        create_network_fn: モデルの作成関数。2個のモデルを返す関数の場合、1個目が訓練用、2個目が推論用。
        nfold: cvの分割数
        train_data_loader: 訓練データの読み込み
        val_data_loader: 検証データの読み込み
        refine_data_loader: Refined Data Augmentation <https://arxiv.org/abs/1909.09148> 用
        models_dir: 保存先ディレクトリ
        model_name_format: モデルのファイル名のフォーマット。{fold}のところに数字が入る。
        skip_if_exists: cv()でモデルが存在するときスキップするならTrue。
        skip_folds: cv()で（skip_if_existsとは関係なく）スキップするfold。[0, nfold)
        epochs: tk.models.fit()のパラメータ
        refine_epochs: refineのエポック数
        refine_lr_factor: refineの学習率の係数。初期学習率×refine_lr_factorがrefine時の学習率になる。
        callbacks: tk.models.fit()のパラメータ
        fit_params: tk.models.fit()のパラメータ
        use_horovod: 推論時にMPIによる分散処理をするか否か。(学習時は常にTrue)
        num_replicas_in_sync: tf.distributeするなら指定する。
        parallel_cv: lgb.cvなどのように全foldまとめて処理するならTrue

    Attributes:
        training_models: 訓練用モデル
        prediction_models: 推論用モデル

    model_name_formatに"{fold}"が含まれない名前を指定した場合、
    cvではなくtrainを使うモードということにする。

    事前にself.training_modelsにモデルがある状態でcvやfitを呼んだ場合は追加学習ということにする。

    """

    def __init__(
        self,
        create_network_fn: typing.Callable[
            [],
            typing.Union[
                tf.keras.models.Model,
                typing.Tuple[tf.keras.models.Model, tf.keras.models.Model],
            ],
        ],
        nfold: int,
        train_data_loader: tk.data.DataLoader,
        val_data_loader: tk.data.DataLoader,
        refine_data_loader: typing.Optional[tk.data.DataLoader] = None,
        *,
        compile_fn: typing.Callable[[tf.keras.models.Model], None] = None,
        epochs: int,
        refine_epochs: int = 50,
        refine_lr_factor: float = 0.003,
        callbacks: typing.List[tf.keras.callbacks.Callback] = None,
        models_dir: tk.typing.PathLike,
        model_name_format: str = "model.fold{fold}.h5",
        skip_if_exists: bool = True,
        skip_folds: typing.Sequence[int] = (),
        fit_params: dict = None,
        use_horovod: bool = False,
        num_replicas_in_sync: int = 1,
        parallel_cv: bool = False,
        on_batch_fn: tk.models.OnBatchFnType = None,
        preprocessors: tk.pipeline.EstimatorListType = None,
        postprocessors: tk.pipeline.EstimatorListType = None,
    ):
        super().__init__(nfold, preprocessors, postprocessors)
        self.create_network_fn = create_network_fn
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.refine_data_loader = refine_data_loader
        self.models_dir = pathlib.Path(models_dir)
        self.model_name_format = model_name_format
        self.skip_if_exists = skip_if_exists
        self.skip_folds = skip_folds
        self.epochs = epochs
        self.compile_fn = compile_fn
        self.refine_epochs = refine_epochs
        self.refine_lr_factor = refine_lr_factor
        self.callbacks = callbacks
        self.fit_params = fit_params
        self.use_horovod = use_horovod
        self.num_replicas_in_sync = num_replicas_in_sync
        self.parallel_cv = parallel_cv
        self.on_batch_fn = on_batch_fn
        self.training_models: typing.List[tf.keras.models.Model] = [None] * nfold
        self.prediction_models: typing.List[tf.keras.models.Model] = [None] * nfold
        if self.parallel_cv:
            assert self.refine_data_loader is None, "NotImplemented"
        if "{fold}" not in self.model_name_format:
            assert nfold == 1

    def _save(self, models_dir: pathlib.Path):
        assert models_dir == self.models_dir

    def _load(self, models_dir: pathlib.Path):
        assert models_dir == self.models_dir
        for fold in range(self.nfold):
            self._load_model(fold)

    def _load_model(self, fold):
        self.create_network(fold)
        model_path = self.models_dir / self.model_name_format.format(fold=fold + 1)
        tk.models.load_weights(self.prediction_models[fold], model_path)

    def _cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType) -> dict:
        assert len(folds) == self.nfold
        if self.parallel_cv:
            return self._parallel_cv(dataset, folds)
        else:
            return self._serial_cv(dataset, folds)

    def _parallel_cv(self, dataset: tk.data.Dataset, folds: tk.validation.FoldsType):
        for fold in range(len(folds)):
            self.create_network(fold)
        assert self.training_models[0] is not None

        inputs = []
        targets = []
        outputs = []
        losses = []
        metrics: dict = {
            n: [] for n in self.training_models[0].metrics_names if n != "loss"
        }
        for i, model in enumerate(self.training_models):
            assert model is not None
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
        model.compile(self.training_models[0].optimizer, loss, list(metrics.values()))
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

    def _predict(self, dataset: tk.data.Dataset, fold: int) -> np.ndarray:
        pred = tk.models.predict(
            self.prediction_models[fold],
            dataset,
            self.val_data_loader,
            use_horovod=self.use_horovod,
            num_replicas_in_sync=self.num_replicas_in_sync,
            on_batch_fn=self.on_batch_fn,
        )
        # メモリを食いがちなので再構築してみる
        self._rebuild_model(fold)
        return pred

    def check(self) -> KerasModel:
        """モデルの動作確認。(KerasModel独自メソッド)

        Returns:
            self

        """
        assert self.nfold >= 1
        self.create_network(fold=0)
        assert self.training_models[0] is not None
        # summary表示
        tk.models.summary(self.training_models[0])
        # グラフを出力
        tk.models.plot(self.training_models[0], self.models_dir / "model.svg")
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
        assert fold in range(self.nfold)
        tk.hvd.barrier()
        tk.log.get(__name__).info(
            f"train: {len(train_set)} samples, "
            f"val: {len(val_set) if val_set is not None else 0} samples, "
            f"batch_size: {self.train_data_loader.batch_size}x{tk.hvd.size() * self.num_replicas_in_sync}"
        )

        model_path = self.models_dir / self.model_name_format.format(fold=fold + 1)

        if self.skip_if_exists and model_path.exists():
            tk.log.get(__name__).info(
                f"Fold {fold + 1}: Loading '{model_path}'... (skip_if_exists)"
            )
            self._load_model(fold)
            trained = False
        elif fold in self.skip_folds and model_path.exists():
            tk.log.get(__name__).info(
                f"Fold {fold + 1}: Loading '{model_path}'... (skip_folds)"
            )
            self._load_model(fold)
            trained = False
        else:
            self.create_network(fold)
            trained = True
            model: tf.keras.models.Model = self.training_models[fold]
            if self.compile_fn is not None:
                self.compile_fn(model)

            if self.refine_data_loader is not None:
                start_lr = tf.keras.backend.get_value(model.optimizer.learning_rate)

            # fit
            tk.hvd.barrier()
            if self.epochs > 0:
                tk.models.fit(
                    model,
                    train_set=train_set,
                    train_data_loader=self.train_data_loader,
                    val_set=val_set,
                    val_data_loader=self.val_data_loader,
                    epochs=self.epochs,
                    callbacks=self.callbacks,
                    num_replicas_in_sync=self.num_replicas_in_sync,
                    **(self.fit_params or {}),
                )

            # refine
            if self.refine_data_loader is not None and self.refine_epochs > 0:
                tk.log.get(__name__).info(
                    f"Fold {fold + 1}: Refining {self.refine_epochs} epochs..."
                )
                tk.models.freeze_layers(model, tf.keras.layers.BatchNormalization)
                if self.compile_fn is None:
                    tk.models.recompile(model)
                else:
                    self.compile_fn(model)
                tf.keras.backend.set_value(
                    model.optimizer.learning_rate, start_lr * self.refine_lr_factor,
                )
                tk.models.fit(
                    model,
                    train_set=train_set,
                    train_data_loader=self.refine_data_loader,
                    val_set=val_set,
                    val_data_loader=self.val_data_loader,
                    epochs=self.refine_epochs,
                    num_replicas_in_sync=self.num_replicas_in_sync,
                )

            tk.models.save(model, model_path)
            del model

        # 訓練データと検証データの評価
        tk.hvd.barrier()
        try:
            self.evaluate(train_set, prefix="", fold=fold)
            if val_set is None:
                return None
            evals = self.evaluate(val_set, prefix="val_", fold=fold)
        except Exception:
            tk.log.get(__name__).warning("evaluate error", exc_info=True)
            evals = {}

        # メモリを食いがちなので再構築してみる
        if trained:
            self._rebuild_model(fold)

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
        assert self.preprocessors is None  # とりあえず未対応
        assert self.postprocessors is None  # とりあえず未対応

        # 未コンパイルならmetricsが無いかもしれないのでcompile
        if self.training_models[fold].optimizer is None:
            assert self.compile_fn is not None
            self.compile_fn(self.training_models[fold])

        evals = tk.models.evaluate(
            self.training_models[fold],
            dataset,
            data_loader=self.val_data_loader,  # DataAugmentation無しで評価
            prefix=prefix,
            use_horovod=self.use_horovod,
            num_replicas_in_sync=self.num_replicas_in_sync,
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
        assert self.preprocessors is None  # とりあえず未対応
        assert self.postprocessors is None  # とりあえず未対応
        return tk.models.predict_flow(
            self.prediction_models[fold],
            dataset,
            self.val_data_loader,
            on_batch_fn=self.on_batch_fn,
            num_replicas_in_sync=self.num_replicas_in_sync,
        )

    def create_network(self, fold: int) -> None:
        """指定foldのモデルの作成。"""
        if self.training_models[fold] is not None:  # 既にあればそれを使う
            assert self.prediction_models[fold] is not None
        else:
            network = self.create_network_fn()
            if not isinstance(network, tuple):
                network = network, network
            self.training_models[fold] = network[0]
            self.prediction_models[fold] = network[1]

    def _rebuild_model(self, fold: int) -> None:
        """メモリ節約のための処理。"""
        self.training_models[fold] = None
        self.prediction_models[fold] = None
        gc.collect()
        self._load_model(fold)
