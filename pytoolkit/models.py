"""Kerasのモデル関連。

Horovodに対応した簡単なwrapperなど。

ただし引数のデフォルトや細かい挙動を変えていたりするので要注意。

"""
from __future__ import annotations

import pathlib
import shutil
import typing

import numpy as np
import tensorflow as tf

import pytoolkit as tk

from . import keras


# モデルの入出力の型
ModelIOType = typing.Union[
    np.ndarray, typing.List[np.ndarray], typing.Dict[str, np.ndarray]
]


def load(
    path,
    custom_objects=None,
    compile=False,  # pylint: disable=redefined-outer-name
    gpus=None,
):
    """モデルの読み込み。"""
    path = pathlib.Path(path)
    with tk.log.trace_scope(f"load({path})"):
        custom_objects = custom_objects.copy() if custom_objects else dict()
        custom_objects.update(tk.get_custom_objects())
        if gpus is not None and gpus > 1:
            with tf.device("/cpu:0"):
                model = keras.models.load_model(
                    str(path), custom_objects=custom_objects, compile=compile
                )
            model, _ = multi_gpu_model(model, batch_size=0, gpus=gpus)
        else:
            model = keras.models.load_model(
                str(path), custom_objects=custom_objects, compile=compile
            )
    return model


def load_weights(model: keras.models.Model, path, by_name=False, skip_not_exist=False):
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
    model: keras.models.Model, path, mode: str = "hdf5", include_optimizer: bool = False
):
    """モデルの保存。

    Args:
        model: モデル
        path (os.PathLike): 保存先。saved_modelの場合はディレクトリ
        mode: "hdf5", "saved_model", "onnx", "tflite"のいずれか
        include_optimizer: HDF5形式で保存する場合にoptimizerを含めるか否か

    """
    assert mode in ("hdf5", "saved_model", "onnx", "tflite")
    path = pathlib.Path(path)
    if tk.hvd.is_master():
        with tk.log.trace_scope(f"save({path})"):
            path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "hdf5":
                model.save(str(path), include_optimizer=include_optimizer)
            elif mode == "saved_model":
                if path.is_dir():
                    shutil.rmtree(path)
                tk.log.get(__name__).info(
                    f"inpus={model.inputs} outputs={model.outputs}"
                )
                tf.saved_model.simple_save(
                    keras.backend.get_session(),
                    str(path),
                    inputs={x.name: x for x in model.inputs},
                    outputs={x.name: x for x in model.outputs},
                )
            elif mode == "onnx":
                import onnxmltools
                import tf2onnx

                input_names = [x.name for x in model.inputs]
                output_names = [x.name for x in model.outputs]
                tk.log.get(__name__).info(
                    f"input_names={input_names} output_names={output_names}"
                )
                onnx_graph = tf2onnx.tfonnx.process_tf_graph(
                    keras.backend.get_session().graph,
                    input_names=input_names,
                    output_names=output_names,
                )
                onnx_model = onnx_graph.make_model("test")
                path.parent.mkdir(parents=True, exist_ok=True)
                onnxmltools.utils.save_model(onnx_model, str(path))
            elif mode == "tflite":
                tk.log.get(__name__).info(
                    f"inpus={model.inputs} outputs={model.outputs}"
                )
                tflite_model = tf.lite.TFLiteConverter.from_session(
                    keras.backend.get_session(),
                    input_tensors=model.inputs,
                    output_tensors=model.outputs,
                ).convert()
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as f:
                    f.write(tflite_model)
            else:
                raise ValueError(f"Invalid save format: {mode}")
    tk.hvd.barrier()


def summary(model: keras.models.Model):
    """summaryを実行するだけ。"""
    model.summary(
        print_fn=tk.log.get(__name__).info if tk.hvd.is_master() else lambda x: None
    )


def plot(
    model: keras.models.Model,
    to_file: str = "model.svg",
    show_shapes: bool = True,
    show_layer_names: bool = True,
    rankdir: str = "TB",
):
    """モデルのグラフのplot。"""
    path = pathlib.Path(to_file)
    if tk.hvd.is_master():
        with tk.log.trace_scope(f"plot({path})"):
            path.parent.mkdir(parents=True, exist_ok=True)
            keras.utils.plot_model(
                model,
                str(path),
                show_shapes=show_shapes,
                show_layer_names=show_layer_names,
                rankdir=rankdir,
            )
    tk.hvd.barrier()


def compile(
    model: keras.models.Model, optimizer, loss, metrics=None, loss_weights=None
):  # pylint: disable=redefined-builtin
    """compileするだけ。"""
    with tk.log.trace_scope("compile"):
        if tk.hvd.initialized():
            optimizer = keras.optimizers.get(optimizer)
            optimizer = tk.hvd.get().DistributedOptimizer(
                optimizer, compression=tk.hvd.get().Compression.fp16
            )
        model.compile(optimizer, loss, metrics, loss_weights=loss_weights)


def fit(
    model: keras.models.Model,
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
    model: keras.models.Model,
    dataset: tk.data.Dataset,
    data_loader: tk.data.DataLoader,
    verbose: int = 1,
    use_horovod: bool = False,
    on_batch_fn: typing.Callable[[keras.models.Model, ModelIOType], ModelIOType] = None,
    flow: bool = False,
    desc: str = "predict",
) -> typing.Union[ModelIOType, typing.Iterator[ModelIOType]]:
    """予測。

    Args:
        model: モデル
        dataset: 予測したい入力データ
        data_loader: データの読み込み
        verbose: プログレスバー(tqdm)を表示するか否か
        use_horovod: MPIによる分散処理をするか否か
        on_batch_fn (callable, optional): モデルとミニバッチ分の入力データを受け取り、予測結果を返す処理。(TTA用)
        flow: 結果をgeneratorで返すならTrue
        desc: flow時のtqdmのdesc

    Returns:
        予測結果。flow=False時はndarray、flow=True時はサンプルごとのgenerator。

    """
    with tk.log.trace_scope("predict"):
        verbose = verbose if tk.hvd.is_master() else 0
        dataset = tk.hvd.split(dataset) if use_horovod else dataset
        iterator = data_loader.iter(dataset)
        if flow:
            assert not use_horovod, "flow=True and use_horovod=True is not supported."
            return _predict_flow(model, iterator, verbose, on_batch_fn, desc)
        else:
            if on_batch_fn is not None:
                values = _predict_flow(model, iterator, verbose, on_batch_fn, desc)
                values = np.array(list(values))
            else:
                values = model.predict(iterator, verbose=verbose)
            values = tk.hvd.allgather(values) if use_horovod else values
            return values


def _predict_flow(model, iterator, verbose, on_batch_fn, desc):
    on_batch_fn = on_batch_fn or _predict_on_batch
    for X, _ in tk.utils.tqdm(
        iterator, desc=desc, total=len(iterator), disable=verbose < 1
    ):
        pred_batch = on_batch_fn(model, X)
        yield from pred_batch


def _predict_on_batch(model: keras.models.Model, X):
    return model.predict_on_batch(X)


def evaluate(
    model: keras.models.Model,
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
    model: keras.models.Model, batch_size: int, gpus: int = None
) -> keras.models.Model:
    """複数GPUでデータ並列するモデルを作成する。

    References:
        - <https://github.com/fchollet/keras/issues/2436>
        - <https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py>

    """
    if gpus is None:
        gpus = tk.dl.get_gpu_count()
        tk.log.get(__name__).info(f"gpu count = {gpus}")
    if gpus <= 1:
        return model, batch_size

    assert isinstance(model.inputs, list)
    assert isinstance(model.outputs, list)

    with tk.log.trace_scope("multi_gpu_model"):
        parallel_model = keras.utils.multi_gpu_model(model, gpus)

        # Model.saveの置き換え
        # https://github.com/fchollet/keras/issues/2436#issuecomment-294243024
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
