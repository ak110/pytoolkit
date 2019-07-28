"""Kerasのモデル関連。

Horovodに対応した簡単なwrapperなど。

ただし引数のデフォルトや細かい挙動を変えていたりするので要注意。

"""
import pathlib
import shutil

import numpy as np
import tensorflow as tf

import pytoolkit as tk

from . import keras
from . import log as tk_log


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


def save(model: keras.models.Model, path, mode="hdf5", include_optimizer=False):
    """モデルの保存。

    Args:
        model (keras.models.Model): モデル
        path (os.PathLike): 保存先。saved_modelの場合はディレクトリ
        format (str, optional): "hdf5", "saved_model", "onnx", "tflite"のいずれか
        include_optimizer (bool, optional): HDF5形式で保存する場合にoptimizerを含めるか否か

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
    to_file="model.svg",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
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


@tk_log.trace()
def compile(
    model: keras.models.Model, optimizer, loss, metrics=None, loss_weights=None
):  # pylint: disable=redefined-builtin
    """compileするだけ。"""
    if tk.hvd.initialized():
        optimizer = keras.optimizers.get(optimizer)
        optimizer = tk.hvd.get().DistributedOptimizer(
            optimizer, compression=tk.hvd.get().Compression.fp16
        )
    model.compile(optimizer, loss, metrics, loss_weights=loss_weights)


@tk_log.trace()
def fit(
    model: keras.models.Model,
    training_data,
    validation_data=None,
    train_preprocessor=None,
    val_preprocessor=None,
    validation_freq="auto",
    class_weight=None,
    batch_size=32,
    epochs=1800,
    callbacks=None,
    verbose=1,
    data_parallel=True,
    initial_epoch=0,
    use_multiprocessing=False,
    workers=1,
    max_queue_size=10,
):
    """独自のtraining loopになる予定の関数。

    Args:
        model: モデル。
        training_data (tk.data.Dataset): 訓練データ。
        validation_data (tk.data.Dataset): 検証データ。Noneなら省略。
        train_preprocessor (tk.data.Preprocessor): 訓練データの前処理
        val_preprocessor (tk.data.Preprocessor): 検証データの前処理
        validation_freq (int or list or "auto"): 検証を行うエポック数の間隔、またはエポック数のリスト。0ならvalidationしない(独自仕様)。"auto"なら適当に決める(独自仕様)。
        class_weight (dict): クラスごとの重みのdict。
        batch_size (int): バッチサイズ。
        epochs (int): エポック数。
        callbacks (list): コールバック。EpochLoggerとErrorOnNaNは自動追加。
        verbose (int): 1ならプログレスバー表示、2ならepoch毎の結果だけ表示。
        data_parallel (bool): DataLoaderで並列化するのか否か。
        initial_epoch (int): 学習を開始するエポック数 - 1。
        use_multiprocessing (bool): Trueならマルチプロセス。
        workers (int): ワーカー数。
        max_queue_size (int): キューの最大サイズ。

    """
    if validation_freq == 0:
        # validation_freq == 0ならvalidationしない(独自仕様)
        validation_freq = None
        validation_data = None
    elif validation_freq == "auto":
        # "auto"なら適当に決める(独自仕様)
        if validation_data is None:
            validation_freq = None
        else:
            # ・sqrt(epochs)回くらいやれば十分？ (指標にも依るが…)
            # ・valがtrainの10%未満くらいなら毎回やっても問題無い
            max_val_per_train = 0.1
            validation_freq = max(
                int(np.sqrt(epochs)),
                int(len(validation_data) / (len(training_data) * max_val_per_train)),
                1,
            )
            # 最後のepochはvalidationしたいので、そこからvalidation_freq毎に。
            validation_freq = list(range(epochs, 0, -validation_freq))

    kwargs = {}
    if validation_freq is not None:
        if tf.__version__ >= "1.14":
            kwargs["validation_freq"] = validation_freq

    train_data_loader = tk.data.DataLoader(
        training_data,
        train_preprocessor,
        batch_size,
        shuffle=True,
        parallel=data_parallel,
        use_horovod=True,
    )
    val_data_loader = (
        tk.data.DataLoader(
            validation_data,
            val_preprocessor,
            batch_size,
            shuffle=True,
            parallel=data_parallel,
            use_horovod=True,
        )
        if validation_data is not None
        else None
    )

    callbacks = (callbacks or []) + [
        tk.callbacks.EpochLogger(),
        tk.callbacks.ErrorOnNaN(),
    ]
    if tk.hvd.initialized() and tk.hvd.size() > 1:
        callbacks.append(tk.hvd.get().callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(tk.hvd.get().callbacks.MetricAverageCallback())

    # TensorFlowのバグ対策
    if tf.__version__ == "1.13.1":
        from tensorflow.python.keras.engine import (  # pylint: disable=no-name-in-module
            training_generator,
        )

        original = training_generator.model_iteration

        def model_iteration_fixed(*args, verbose=0, **kwargs):
            return original(*args, verbose=verbose, **kwargs)

        training_generator.model_iteration = model_iteration_fixed
    try:
        model.fit_generator(
            train_data_loader,
            validation_data=val_data_loader,
            class_weight=class_weight,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose if tk.hvd.is_master() else 0,
            initial_epoch=initial_epoch,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            max_queue_size=max_queue_size,
            **kwargs,
        )
    finally:
        if tf.__version__ == "1.13.1":
            training_generator.model_iteration = original

    # DataLoaderの処理時間を表示
    tk.log.get(__name__).info(
        f"train_data_loader: {train_data_loader.seconds_per_step * 1000:4.0f}ms/step"
    )
    if val_data_loader is not None:
        tk.log.get(__name__).info(
            f"val_data_loader:   {train_data_loader.seconds_per_step * 1000:4.0f}ms/step"
        )


@tk_log.trace()
def predict(
    model: keras.models.Model,
    dataset,
    preprocessor=None,
    batch_size=32,
    verbose=1,
    use_horovod=False,
    on_batch_fn=None,
    flow=False,
    desc="predict",
):
    """予測。

    Args:
        model: モデル。
        dataset: 予測したい入力データ。
        preprocessor (tk.data.Preprocessor): 前処理
        batch_size (int): バッチサイズ
        verbose (int): プログレスバー(tqdm)を表示するか否か。
        use_horovod (bool): MPIによる分散処理をするか否か。
        on_batch_fn (callable, optional): モデルとミニバッチ分の入力データを受け取り、予測結果を返す処理。(TTA用)
        flow (bool): 結果をgeneratorで返すならTrue。
        desc (str): flow時のtqdmのdesc。

    Returns:
        np.ndarray or generator: 予測結果。flow=True時はサンプルごとのgenerator。

    """
    if on_batch_fn is not None and not flow:
        return np.array(
            list(
                predict(
                    model=model,
                    dataset=dataset,
                    preprocessor=preprocessor,
                    batch_size=batch_size,
                    verbose=verbose,
                    use_horovod=use_horovod,
                    on_batch_fn=on_batch_fn,
                    flow=True,
                    desc=desc,
                )
            )
        )

    dataset = tk.hvd.split(dataset) if use_horovod else dataset
    data_loader = tk.data.DataLoader(dataset, preprocessor, batch_size)
    if flow:
        assert not use_horovod, "flow=True and use_horovod=True is not supported."
        return _predict_flow(model, data_loader, verbose, on_batch_fn, desc)
    else:
        values = model.predict_generator(
            data_loader, verbose=verbose if tk.hvd.is_master() else 0
        )
        values = tk.hvd.allgather(values) if use_horovod else values
        return values


def _predict_flow(model, data_loader, verbose, on_batch_fn, desc):
    on_batch_fn = on_batch_fn or _predict_on_batch
    for X, _ in tk.utils.tqdm(
        data_loader, desc=desc, total=len(data_loader), disable=verbose < 1
    ):
        pred_batch = on_batch_fn(model, X)
        yield from pred_batch


def _predict_on_batch(model: keras.models.Model, X):
    return model.predict_on_batch(X)


@tk_log.trace()
def evaluate(
    model: keras.models.Model,
    dataset,
    preprocessor=None,
    batch_size=32,
    verbose=1,
    use_horovod=False,
):
    """評価。

    Args:
        model: モデル。
        dataset (tk.data.Dataset): データ。
        preprocessor (tk.data.Preprocessor): 前処理
        verbose (int): 1ならプログレスバー表示。
        use_horovod (bool): MPIによる分散処理をするか否か。

    Returns:
        dict: metricsの文字列と値のdict

    """
    dataset = tk.hvd.split(dataset) if use_horovod else dataset
    data_loader = tk.data.DataLoader(dataset, preprocessor, batch_size)
    values = model.evaluate_generator(
        data_loader, verbose=verbose if tk.hvd.is_master() else 0
    )
    values = tk.hvd.allreduce(values) if use_horovod else values
    if len(model.metrics_names) == 1:
        evals = {model.metrics_names[0]: values}
    else:
        evals = dict(zip(model.metrics_names, values))
    return evals


@tk_log.trace()
def multi_gpu_model(model, batch_size, gpus=None):
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
