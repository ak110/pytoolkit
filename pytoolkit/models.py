"""Kerasのモデル関連。

Horovodに対応した簡単なwrapperなど。

ただし引数のデフォルトや細かい挙動を変えていたりするので要注意。

"""
from __future__ import annotations

import pathlib
import typing

import numpy as np
import tensorflow as tf

import pytoolkit as tk

# モデルの入出力の型
ModelIOType = typing.Union[
    np.ndarray, typing.List[np.ndarray], typing.Dict[str, np.ndarray]
]
# predictで使う型
OnBatchFnType = typing.Callable[[tf.keras.models.Model, ModelIOType], ModelIOType]
# compileで使う型
OptimizerType = typing.Union[str, tf.keras.optimizers.Optimizer]
LossType = typing.Union[
    str, tf.keras.losses.Loss, typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
]
MetricType = typing.Union[
    str, tf.keras.metrics.Metric, typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
]
MetricsType = typing.List[MetricType]


def load(
    path: tk.typing.PathLike,
    custom_objects: dict = None,
    compile: bool = False,  # pylint: disable=redefined-outer-name
):
    """モデルの読み込み。"""
    with tk.log.trace_scope(f"load({path})"):
        model = tf.keras.models.load_model(
            str(path), custom_objects=custom_objects, compile=compile
        )
    return model


def load_weights(
    model: tf.keras.models.Model,
    path: tk.typing.PathLike,
    by_name: bool = False,
    skip_not_exist: bool = False,
    verbose: bool = True,
):
    """モデルの重みの読み込み。"""
    path = pathlib.Path(path)
    if path.exists():
        with tk.log.trace_scope(f"load_weights({path})"):
            if verbose:
                old_weights = model.get_weights()
            model.load_weights(str(path), by_name=by_name)
            if verbose:
                new_weights = model.get_weights()
                changed_params = np.sum(
                    [
                        np.sum(np.not_equal(w1, w2))
                        for w1, w2 in zip(old_weights, new_weights)
                    ]
                )
                num_params = np.sum([w.size for w in new_weights])
                tk.log.get(__name__).info(
                    f"{changed_params:,} params chagnged. ({changed_params / num_params:.1%})"
                )
    elif skip_not_exist:
        tk.log.get(__name__).info(f"{path} is not found.")
    else:
        raise RuntimeError(f"{path} is not found.")


def save(
    model: tf.keras.models.Model,
    path: tk.typing.PathLike,
    mode: str = "hdf5",
    include_optimizer: bool = False,
):
    """モデルの保存。

    Args:
        model: モデル
        path: 保存先。saved_modelの場合はディレクトリ
        mode: "hdf5", "saved_model", "onnx", "tflite"のいずれか
        include_optimizer: HDF5形式で保存する場合にoptimizerを含めるか否か

    """
    assert mode in ("hdf5", "saved_model", "onnx", "tflite")
    path = pathlib.Path(path)
    if tk.hvd.is_master():
        with tk.log.trace_scope(f"save({path})"):
            path.parent.mkdir(parents=True, exist_ok=True)
            if mode in ("hdf5", "saved_model"):
                model.save(
                    str(path),
                    overwrite=True,
                    include_optimizer=include_optimizer,
                    save_format={"hdf5": "h5", "saved_model": "tf"}[mode],
                )
            elif mode == "onnx":
                import onnxmltools
                import keras2onnx

                onnx_model = keras2onnx.convert_keras(model, model.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                onnxmltools.utils.save_model(onnx_model, str(path))
            elif mode == "tflite":
                tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as f:
                    f.write(tflite_model)
            else:
                raise ValueError(f"Invalid save format: {mode}")
    tk.hvd.barrier()


def summary(model: tf.keras.models.Model):
    """summaryを実行するだけ。"""
    model.summary(
        print_fn=tk.log.get(__name__).info if tk.hvd.is_master() else lambda x: None
    )


def plot(
    model: tf.keras.models.Model,
    to_file: tk.typing.PathLike = "model.svg",
    show_shapes: bool = True,
    show_layer_names: bool = True,
    rankdir: str = "TB",
):
    """モデルのグラフのplot。"""
    path = pathlib.Path(to_file)
    if tk.hvd.is_master():
        with tk.log.trace_scope(f"plot({path})"):
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                tf.keras.utils.plot_model(
                    model,
                    str(path),
                    show_shapes=show_shapes,
                    show_layer_names=show_layer_names,
                    rankdir=rankdir,
                )
            except ValueError:
                pass  # "Cannot embed the 'svg' image format" (tf >= 1.14)
    tk.hvd.barrier()


def compile(
    model: tf.keras.models.Model,
    optimizer: OptimizerType,
    loss: LossType = None,
    metrics: MetricsType = None,
    experimental_run_tf_function: bool = None,
):  # pylint: disable=redefined-builtin
    """compileするだけ。"""
    with tk.log.trace_scope("compile"):
        if tk.hvd.initialized():
            optimizer = tf.keras.optimizers.get(optimizer)
            optimizer = tk.hvd.get().DistributedOptimizer(
                optimizer, compression=tk.hvd.get().Compression.fp16
            )
            # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
            # uses hvd.DistributedOptimizer() to compute gradients.
            if experimental_run_tf_function is None:
                experimental_run_tf_function = False
        else:
            if experimental_run_tf_function is None:
                experimental_run_tf_function = True
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            experimental_run_tf_function=experimental_run_tf_function,
        )


def recompile(model: tf.keras.models.Model):
    """optimizerなどを再利用してコンパイル。"""
    with tk.log.trace_scope("recompile"):
        # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
        # uses hvd.DistributedOptimizer() to compute gradients.
        model.compile(
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=model.metrics,
            experimental_run_tf_function=False,
        )


def fit(
    model: tf.keras.models.Model,
    train_set: tk.data.Dataset,
    train_data_loader: tk.data.DataLoader,
    val_set: tk.data.Dataset = None,
    val_data_loader: tk.data.DataLoader = None,
    validation_freq: typing.Union[int, typing.Sequence[int], str, None] = "auto",
    class_weight: dict = None,
    epochs: int = 1800,
    callbacks: list = None,
    verbose: int = 1,
    initial_epoch: int = 0,
    use_multiprocessing: bool = False,
    workers: int = 1,
    max_queue_size: int = 10,
    num_replicas_in_sync: int = 1,
):
    """独自のtraining loopになる予定の関数。

    Args:
        model: モデル
        train_set: 訓練データ
        train_data_loader: 訓練データの読み込み
        val_set: 検証データ。Noneなら省略。
        val_data_loader: 検証データの読み込み
        validation_freq: 検証を行うエポック数の間隔、またはエポック数のリスト。0ならvalidationしない(独自仕様)。"auto"なら適当に決める(独自仕様)。
        class_weight: クラスごとの重みのdict
        epochs: エポック数
        callbacks: コールバック。EpochLoggerとErrorOnNaNとhorovod関連は自動追加。
        verbose: 1ならプログレスバー表示、2ならepoch毎の結果だけ表示。
        initial_epoch: 学習を開始するエポック数 - 1
        use_multiprocessing: Trueならマルチプロセス
        workers: ワーカー数
        max_queue_size: キューの最大サイズ
        num_replicas_in_sync: tf.distribute使用時の並列数(バッチサイズに掛け算する)

    """
    if validation_freq == 0:
        # validation_freq == 0ならvalidationしない(独自仕様)
        validation_freq = None
        val_set = None
        val_data_loader = None
    elif validation_freq == "auto":
        # "auto"なら適当に決める(独自仕様)
        validation_freq = make_validation_freq(
            validation_freq, epochs, train_set, val_set
        )
    if val_set is not None:
        assert val_data_loader is not None

    train_iterator = train_data_loader.iter(
        train_set,
        shuffle=True,
        use_horovod=True,
        num_replicas_in_sync=num_replicas_in_sync,
    )
    val_iterator = (
        val_data_loader.iter(
            val_set,
            shuffle=True,
            use_horovod=True,
            num_replicas_in_sync=num_replicas_in_sync,
        )
        if val_set is not None and val_data_loader is not None
        else None
    )

    callbacks = make_callbacks(callbacks, training=True)

    fit_kwargs = {}
    if validation_freq is not None:
        fit_kwargs["validation_freq"] = validation_freq

    with tk.log.trace_scope("fit"):
        model.fit(
            train_iterator.ds,
            steps_per_epoch=train_iterator.steps,
            validation_data=val_iterator.ds if val_iterator is not None else None,
            validation_steps=val_iterator.steps if val_iterator is not None else None,
            class_weight=class_weight,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose if tk.hvd.is_master() else 0,
            initial_epoch=initial_epoch,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            max_queue_size=max_queue_size,
            **fit_kwargs,
        )


def make_validation_freq(
    validation_freq, epochs, train_set, val_set, max_val_per_train=0.1
):
    """validation_freqをほどよい感じに作成する。"""
    if val_set is None:
        return None
    # sqrt(epochs)回くらいやれば十分？ (指標にも依るが…)
    # valがtrainの10%未満くらいなら毎回やっても問題無い
    validation_freq = max(
        int(np.sqrt(epochs)),
        int(len(val_set) / (len(train_set) * max_val_per_train)),
        1,
    )
    # 最低でも10回くらいはやりたい
    validation_freq = min(validation_freq, max(1, epochs // 10))
    # 最後のepochはvalidationしたいので、そこからvalidation_freq毎に。
    validation_list = list(range(epochs, 0, -validation_freq))
    # あまり早いepochではやらない
    if len(validation_list) >= 2 and validation_list[0] < validation_freq:
        validation_list = validation_list[1:]
    return validation_list


def make_callbacks(callbacks, training: bool) -> list:
    """callbacksをいい感じにする。"""
    callbacks = (callbacks or []).copy()
    if training:
        callbacks.append(tk.callbacks.EpochLogger())
        callbacks.append(tk.callbacks.ErrorOnNaN())
    if tk.hvd.initialized() and tk.hvd.size() > 1:
        callbacks.append(tk.hvd.get().callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(tk.hvd.get().callbacks.MetricAverageCallback())
    return callbacks


def predict(
    model: tf.keras.models.Model,
    dataset: tk.data.Dataset,
    data_loader: tk.data.DataLoader,
    callbacks: list = None,
    verbose: int = 1,
    use_horovod: bool = False,
    on_batch_fn: OnBatchFnType = None,
    num_replicas_in_sync: int = 1,
) -> ModelIOType:
    """予測。

    Args:
        model: モデル
        dataset: 予測したい入力データ
        data_loader: データの読み込み
        callbacks: コールバック
        verbose: プログレスバーを表示するか否か
        use_horovod: MPIによる分散処理をするか否か
        on_batch_fn: モデルとミニバッチ分の入力データを受け取り、予測結果を返す処理。(TTA用)
        flow: 結果をgeneratorで返すならTrue
        desc: flow時のtqdmのdesc
        num_replicas_in_sync: tf.distribute使用時の並列数(バッチサイズに掛け算する)

    Returns:
        予測結果。

    """
    with tk.log.trace_scope("predict"):
        verbose = verbose if tk.hvd.is_master() else 0
        callbacks = make_callbacks(callbacks, training=False)
        dataset = tk.hvd.split(dataset) if use_horovod else dataset
        iterator = data_loader.iter(dataset, num_replicas_in_sync=num_replicas_in_sync)
        if on_batch_fn is not None:
            values = _predict_flow(
                model=model,
                iterator=iterator,
                callbacks=callbacks,
                verbose=verbose,
                on_batch_fn=on_batch_fn,
                desc="predict",
            )
            values = np.array(list(values))
        else:
            values = model.predict(
                iterator.ds, steps=iterator.steps, verbose=verbose, callbacks=callbacks,
            )
        values = tk.hvd.allgather(values) if use_horovod else values
        return values


def predict_flow(
    model: tf.keras.models.Model,
    dataset: tk.data.Dataset,
    data_loader: tk.data.DataLoader,
    callbacks: list = None,
    verbose: int = 1,
    on_batch_fn: OnBatchFnType = None,
    desc: str = "predict",
    num_replicas_in_sync: int = 1,
) -> typing.Iterator[ModelIOType]:
    """予測。

    Args:
        model: モデル
        dataset: 予測したい入力データ
        data_loader: データの読み込み
        callbacks: コールバック
        verbose: プログレスバー(tqdm)を表示するか否か
        on_batch_fn: モデルとミニバッチ分の入力データを受け取り、予測結果を返す処理。(TTA用)
        flow: 結果をgeneratorで返すならTrue
        desc: flow時のtqdmのdesc
        num_replicas_in_sync: tf.distribute使用時の並列数(バッチサイズに掛け算する)

    Returns:
        予測結果。サンプルごとのgenerator。

    """
    with tk.log.trace_scope("predict"):
        callbacks = make_callbacks(callbacks, training=False)
        iterator = data_loader.iter(dataset, num_replicas_in_sync=num_replicas_in_sync)
        return _predict_flow(
            model=model,
            iterator=iterator,
            callbacks=callbacks,
            verbose=verbose,
            on_batch_fn=on_batch_fn,
            desc=desc,
        )


def _predict_flow(
    model: tf.keras.models.Model,
    iterator: tk.data.Iterator,
    callbacks: list,
    verbose: int,
    on_batch_fn: OnBatchFnType = None,
    desc: str = "predict",
):
    on_batch_fn = on_batch_fn or _predict_on_batch
    for cb in callbacks:
        cb.on_predict_begin()
    batch = 0
    for X, _ in tk.utils.tqdm(
        iterator.ds, desc=desc, total=iterator.steps, disable=verbose < 1
    ):
        for cb in callbacks:
            cb.on_predict_batch_begin(batch)
        pred_batch = on_batch_fn(model, X)
        for cb in callbacks:
            cb.on_predict_batch_end(batch)
        yield from pred_batch
        batch += 1
    for cb in callbacks:
        cb.on_predict_end()


def _predict_on_batch(model: tf.keras.models.Model, X):
    return model.predict_on_batch(X)


def evaluate(
    model: tf.keras.models.Model,
    dataset: tk.data.Dataset,
    data_loader: tk.data.DataLoader,
    callbacks: list = None,
    verbose: int = 1,
    prefix: str = "",
    use_horovod: bool = False,
    num_replicas_in_sync: int = 1,
) -> typing.Dict[str, float]:
    """評価。

    Args:
        model: モデル。
        dataset: データ。
        data_loader: データの読み込み
        callbacks: コールバック
        verbose: 1ならプログレスバー表示
        prefix: メトリクス名の接頭文字列
        use_horovod: MPIによる分散処理をするか否か
        num_replicas_in_sync: tf.distribute使用時の並列数(バッチサイズに掛け算する)

    Returns:
        メトリクス名と値のdict

    """
    with tk.log.trace_scope("evaluate"):
        verbose = verbose if tk.hvd.is_master() else 0
        callbacks = make_callbacks(callbacks, training=False)
        dataset = tk.hvd.split(dataset) if use_horovod else dataset
        iterator = data_loader.iter(dataset, num_replicas_in_sync=num_replicas_in_sync)
        values = model.evaluate(
            iterator.ds, steps=iterator.steps, verbose=verbose, callbacks=callbacks,
        )
        values = tk.hvd.allreduce(values) if use_horovod else values
        if len(model.metrics_names) == 1:
            evals = {prefix + model.metrics_names[0]: values}
        else:
            evals = dict(zip([prefix + n for n in model.metrics_names], values))
        return evals


def freeze_layers(
    model: typing.Union[tf.keras.models.Model, tf.keras.layers.Layer], layer_class: type
):
    """指定したレイヤーをfreezeする。"""
    for layer in model.layers:
        if isinstance(layer, layer_class):
            typing.cast(tf.keras.layers.Layer, layer).trainable = False
        if hasattr(layer, "layers") and len(layer.layers) > 0:
            freeze_layers(layer, layer_class)


def predict_on_batch_augmented(
    model: tf.keras.models.Model,
    X_batch: np.ndarray,
    flip: typing.Tuple[bool, bool] = (False, True),
    crop_size: typing.Tuple[int, int] = (3, 3),
    padding_size: typing.Tuple[int, int] = (32, 32),
    padding_mode: str = "edge",
) -> typing.List[np.ndarray]:
    """ミニバッチ1個分の予測処理＆TTA。

    Args:
        model: モデル。
        X_batch: データ。
        flip: 水平/垂直方向の反転を行うか否か。(v, h)
        crop_size: 縦横のcropのパターンの数。(v, h)
        padding_size: crop前にパディングするサイズ。(v, h)
        padding_mode: パディングの種類。(np.padのmode)

    Returns:
        予測結果のリスト。

    """
    shape = X_batch.shape
    X_batch = np.pad(
        X_batch,
        (
            (0, 0),
            (padding_size[0], padding_size[0]),
            (padding_size[1], padding_size[1]),
            (0, 0),
        ),
        mode=padding_mode,
    )
    X_batch2: list = []
    for y in np.linspace(0, padding_size[0] * 2, crop_size[0], dtype=np.int32):
        for x in np.linspace(0, padding_size[1] * 2, crop_size[1], dtype=np.int32):
            X = X_batch[:, x : x + shape[1], y : y + shape[2], :]
            X_batch2.append(X)
            if flip[0]:
                X_batch2.append(X[:, ::-1, :, :])
            if flip[1]:
                X_batch2.append(X[:, :, ::-1, :])
            if flip[0] and flip[1]:
                X_batch2.append(X[:, ::-1, ::-1, :])
    result = model.predict(np.concatenate(X_batch2, axis=0), verbose=0)
    result = result.reshape((len(X_batch2), len(X_batch)) + result.shape[1:])
    return result
