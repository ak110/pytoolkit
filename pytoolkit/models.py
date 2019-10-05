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


def load(
    path: tk.typing.PathLike,
    custom_objects: dict = None,
    compile: bool = False,  # pylint: disable=redefined-outer-name
    gpus: int = None,
):
    """モデルの読み込み。"""
    path = pathlib.Path(path)
    with tk.log.trace_scope(f"load({path})"):
        custom_objects = custom_objects.copy() if custom_objects else {}
        custom_objects.update(tk.get_custom_objects())
        if gpus is not None and gpus > 1:
            with tf.device("/cpu:0"):
                model = tf.keras.models.load_model(
                    str(path), custom_objects=custom_objects, compile=compile
                )
            model, _ = multi_gpu_model(model, batch_size=0, gpus=gpus)
        else:
            model = tf.keras.models.load_model(
                str(path), custom_objects=custom_objects, compile=compile
            )
    return model


def load_weights(
    model: tf.keras.models.Model,
    path: tk.typing.PathLike,
    by_name: bool = False,
    skip_not_exist: bool = False,
):
    """モデルの重みの読み込み。"""
    path = pathlib.Path(path)
    if path.exists():
        with tk.log.trace_scope(f"load_weights({path})"):
            model.load_weights(str(path), by_name=by_name)
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
            tf.keras.utils.plot_model(
                model,
                str(path),
                show_shapes=show_shapes,
                show_layer_names=show_layer_names,
                rankdir=rankdir,
            )
    tk.hvd.barrier()


def compile(
    model: tf.keras.models.Model, optimizer, loss, metrics=None, loss_weights=None
):  # pylint: disable=redefined-builtin
    """compileするだけ。"""
    with tk.log.trace_scope("compile"):
        kwargs = {}
        if tk.hvd.initialized():
            optimizer = tf.keras.optimizers.get(optimizer)
            optimizer = tk.hvd.get().DistributedOptimizer(
                optimizer, compression=tk.hvd.get().Compression.fp16
            )
            # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
            # uses hvd.DistributedOptimizer() to compute gradients.
            kwargs["experimental_run_tf_function"] = False
        model.compile(optimizer, loss, metrics, loss_weights=loss_weights, **kwargs)


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
        callbacks: コールバック。EpochLoggerとErrorOnNaNは自動追加。
        verbose: 1ならプログレスバー表示、2ならepoch毎の結果だけ表示。
        initial_epoch: 学習を開始するエポック数 - 1
        use_multiprocessing: Trueならマルチプロセス
        workers: ワーカー数
        max_queue_size: キューの最大サイズ

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
    assert (val_set is None) == (val_data_loader is None)

    train_iterator = train_data_loader.iter(train_set, shuffle=True, use_horovod=True)
    val_iterator = (
        val_data_loader.iter(val_set, shuffle=True, use_horovod=True)
        if val_set is not None and val_data_loader is not None
        else None
    )

    callbacks = make_callbacks(callbacks)

    fit_kwargs = {}
    if validation_freq is not None:
        fit_kwargs["validation_freq"] = validation_freq

    with tk.log.trace_scope("fit"):
        model.fit(
            train_iterator,
            validation_data=val_iterator,
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

    # DataLoaderの処理時間を表示
    tk.log.get(__name__).info(
        f"train_iterator: {train_iterator.seconds_per_step * 1000:4.0f}ms/step"
    )
    if val_iterator is not None:
        tk.log.get(__name__).info(
            f"val_iterator:   {val_iterator.seconds_per_step * 1000:4.0f}ms/step"
        )


def make_validation_freq(
    validation_freq, epochs, train_set, val_set, max_val_per_train=0.1
):
    """validation_freqをほどよい感じに作成する。"""
    if val_set is None:
        return None
    # ・sqrt(epochs)回くらいやれば十分？ (指標にも依るが…)
    # ・valがtrainの10%未満くらいなら毎回やっても問題無い
    validation_freq = max(
        int(np.sqrt(epochs)),
        int(len(val_set) / (len(train_set) * max_val_per_train)),
        1,
    )
    # 最後のepochはvalidationしたいので、そこからvalidation_freq毎に。
    validation_freq = list(range(epochs, 0, -validation_freq))
    return validation_freq


def make_callbacks(callbacks):
    """callbacksをいい感じにする。"""
    callbacks = (callbacks or []) + [
        tk.callbacks.EpochLogger(),
        tk.callbacks.ErrorOnNaN(),
    ]
    if tk.hvd.initialized() and tk.hvd.size() > 1:
        callbacks.append(tk.hvd.get().callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(tk.hvd.get().callbacks.MetricAverageCallback())
    return callbacks


def predict(
    model: tf.keras.models.Model,
    dataset: tk.data.Dataset,
    data_loader: tk.data.DataLoader,
    verbose: int = 1,
    use_horovod: bool = False,
    on_batch_fn: OnBatchFnType = None,
) -> ModelIOType:
    """予測。

    Args:
        model: モデル
        dataset: 予測したい入力データ
        data_loader: データの読み込み
        verbose: プログレスバー(tqdm)を表示するか否か
        use_horovod: MPIによる分散処理をするか否か
        on_batch_fn: モデルとミニバッチ分の入力データを受け取り、予測結果を返す処理。(TTA用)
        flow: 結果をgeneratorで返すならTrue
        desc: flow時のtqdmのdesc

    Returns:
        予測結果。

    """
    with tk.log.trace_scope("predict"):
        verbose = verbose if tk.hvd.is_master() else 0
        dataset = tk.hvd.split(dataset) if use_horovod else dataset
        iterator = data_loader.iter(dataset)
        if on_batch_fn is not None:
            values = _predict_flow(
                model, iterator, verbose, on_batch_fn, desc="predict"
            )
            values = np.array(list(values))
        else:
            values = model.predict(iterator, verbose=verbose)
        values = tk.hvd.allgather(values) if use_horovod else values
        return values


def predict_flow(
    model: tf.keras.models.Model,
    dataset: tk.data.Dataset,
    data_loader: tk.data.DataLoader,
    verbose: int = 1,
    on_batch_fn: OnBatchFnType = None,
    desc: str = "predict",
) -> typing.Iterator[ModelIOType]:
    """予測。

    Args:
        model: モデル
        dataset: 予測したい入力データ
        data_loader: データの読み込み
        verbose: プログレスバー(tqdm)を表示するか否か
        on_batch_fn: モデルとミニバッチ分の入力データを受け取り、予測結果を返す処理。(TTA用)
        flow: 結果をgeneratorで返すならTrue
        desc: flow時のtqdmのdesc

    Returns:
        予測結果。サンプルごとのgenerator。

    """
    with tk.log.trace_scope("predict"):
        iterator = data_loader.iter(dataset)
        return _predict_flow(model, iterator, verbose, on_batch_fn, desc)


def _predict_flow(model, iterator, verbose, on_batch_fn, desc):
    on_batch_fn = on_batch_fn or _predict_on_batch
    for X, _ in tk.utils.tqdm(
        iterator, desc=desc, total=len(iterator), disable=verbose < 1
    ):
        pred_batch = on_batch_fn(model, X)
        yield from pred_batch


def _predict_on_batch(model: tf.keras.models.Model, X):
    return model.predict_on_batch(X)


def evaluate(
    model: tf.keras.models.Model,
    dataset: tk.data.Dataset,
    data_loader: tk.data.DataLoader,
    verbose: int = 1,
    prefix: str = "",
    use_horovod: bool = False,
) -> typing.Dict[str, float]:
    """評価。

    Args:
        model: モデル。
        dataset: データ。
        data_loader: データの読み込み
        verbose: 1ならプログレスバー表示。
        prefix: メトリクス名の接頭文字列。
        use_horovod: MPIによる分散処理をするか否か。

    Returns:
        メトリクス名と値のdict

    """
    with tk.log.trace_scope("evaluate"):
        dataset = tk.hvd.split(dataset) if use_horovod else dataset
        iterator = data_loader.iter(dataset)
        values = model.evaluate(iterator, verbose=verbose if tk.hvd.is_master() else 0)
        values = tk.hvd.allreduce(values) if use_horovod else values
        if len(model.metrics_names) == 1:
            evals = {prefix + model.metrics_names[0]: values}
        else:
            evals = dict(zip([prefix + n for n in model.metrics_names], values))
        return evals


def multi_gpu_model(
    model: tf.keras.models.Model, batch_size: int, gpus: int = None
) -> tf.keras.models.Model:
    """複数GPUでデータ並列するモデルを作成する。

    References:
        - <https://github.com/fchollet/tf.keras/issues/2436>
        - <https://github.com/kuza55/tf.keras-extras/blob/master/utils/multi_gpu.py>

    """
    if gpus is None:
        gpus = tk.dl.get_gpu_count()
        tk.log.get(__name__).info(f"gpu count = {gpus}")
    if gpus <= 1:
        return model, batch_size

    assert isinstance(model.inputs, list)
    assert isinstance(model.outputs, list)

    with tk.log.trace_scope("multi_gpu_model"):
        parallel_model = tf.keras.utils.multi_gpu_model(model, gpus)

        # Model.saveの置き換え
        # https://github.com/fchollet/tf.keras/issues/2436#issuecomment-294243024
        def _save(self_, *args, **kargs):
            assert self_ is not None  # noqa
            model.save(*args, **kargs)

        def _save_weights(self_, *args, **kargs):
            assert self_ is not None  # noqa
            model.save_weights(*args, **kargs)

        parallel_model.save = type(model.save)(_save, parallel_model)
        parallel_model.save_weights = type(model.save_weights)(
            _save_weights, parallel_model
        )

        return parallel_model, batch_size * gpus
