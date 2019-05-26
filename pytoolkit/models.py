"""Kerasのモデル関連。

Horovodに対応した簡単なwrapperなど。

ただし引数のデフォルトや細かい挙動を変えていたりするので要注意。

"""
import pathlib

import numpy as np
import tensorflow as tf

from .. import pytoolkit as tk
from . import keras

_logger = tk.log.get(__name__)


def load(path, custom_objects=None, compile=False, gpus=None) -> keras.models.Model:  # pylint: disable=redefined-outer-name
    """モデルの読み込み。"""
    path = pathlib.Path(path)
    with tk.log.trace_scope(f'load({path})'):
        custom_objects = custom_objects.copy() if custom_objects else dict()
        custom_objects.update(tk.get_custom_objects())
        if gpus is not None and gpus > 1:
            with tf.device('/cpu:0'):
                model = keras.models.load_model(str(path), custom_objects=custom_objects, compile=compile)
            model, _ = multi_gpu_model(model, batch_size=0, gpus=gpus)
        else:
            model = keras.models.load_model(str(path), custom_objects=custom_objects, compile=compile)
    return model


def load_weights(model: keras.models.Model, path, by_name=False, skip_not_exist=False):
    """モデルの重みの読み込み。"""
    path = pathlib.Path(path)
    if path.exists():
        with tk.log.trace_scope(f'load_weights({path})'):
            model.load_weights(str(path), by_name=by_name)
    elif skip_not_exist:
        tk.log.get(__name__).info(f'{path} is not found.')
    else:
        raise RuntimeError(f'{path} is not found.')


def save(model: keras.models.Model, path, include_optimizer=False):
    """モデルの保存。"""
    path = pathlib.Path(path)
    if tk.hvd.is_master():
        with tk.log.trace_scope(f'save({path})'):
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(path), include_optimizer=include_optimizer)
    tk.hvd.barrier()


def summary(model: keras.models.Model):
    """summaryを実行するだけ。"""
    model.summary(print_fn=tk.log.get(__name__).info if tk.hvd.is_master() else lambda x: None)


def plot(model: keras.models.Model, to_file='model.svg', show_shapes=True, show_layer_names=True, rankdir='TB'):
    """モデルのグラフのplot。"""
    path = pathlib.Path(to_file)
    if tk.hvd.is_master():
        with tk.log.trace_scope(f'plot({path})'):
            path.parent.mkdir(parents=True, exist_ok=True)
            keras.utils.plot_model(model, str(path), show_shapes=show_shapes, show_layer_names=show_layer_names, rankdir=rankdir)
    tk.hvd.barrier()


@tk.log.trace()
def compile(model: keras.models.Model, optimizer, loss, metrics=None, loss_weights=None):  # pylint: disable=redefined-builtin
    """compileするだけ。"""
    if tk.hvd.initialized():
        optimizer = keras.optimizers.get(optimizer)
        optimizer = tk.hvd.get().DistributedOptimizer(optimizer, compression=tk.hvd.get().Compression.fp16)
    model.compile(optimizer, loss, metrics, loss_weights=loss_weights)


@tk.log.trace()
def fit(model: keras.models.Model,
        training_data: tk.data.Dataset,
        validation_data: tk.data.Dataset = None,
        validation_freq: int = 1,
        class_weight=None,
        batch_size=32, epochs=1800,
        callbacks=None, verbose=1,
        data_parallel=True,
        initial_epoch=0,
        use_multiprocessing=False, workers=1, max_queue_size=10,
        warmup=True):
    """独自のtraining loopになる予定の関数。

    Args:
        model: モデル。
        training_data (tk.data.Dataset): 訓練データ。
        validation_data (tk.data.Dataset): 検証データ。Noneなら省略。
        validation_freq (int or list): 検証を行うエポック数の間隔、またはエポック数のリスト。0ならvalidationしない(独自仕様)。
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
        warmup (bool): HorovodのLearningRateWarmupCallbackを使うか否か。

    """
    # validation_freq == 0ならvalidationしない(独自仕様)
    if validation_freq == 0:
        validation_data = None

    train_data_loader = tk.data.DataLoader(training_data, batch_size, shuffle=True, parallel=data_parallel, use_horovod=True)
    val_data_loader = tk.data.DataLoader(validation_data, batch_size, shuffle=True, parallel=data_parallel, use_horovod=True) if validation_data is not None else None

    callbacks = (callbacks or []) + [
        tk.callbacks.EpochLogger(),
        tk.callbacks.ErrorOnNaN(),
    ]
    if tk.hvd.initialized():
        callbacks.append(tk.hvd.get().callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(tk.hvd.get().callbacks.MetricAverageCallback())
        if warmup:
            callbacks.append(tk.hvd.get().callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))

    # TODO: validation_freqはTensorFlowに合わせて対応予定

    # TensorFlowのバグ対策
    if tf.__version__ == '1.13.1':
        from tensorflow.python.keras.engine import training_generator  # pylint: disable=no-name-in-module
        original = training_generator.model_iteration
        training_generator.model_iteration = lambda *args, verbose=0, **kwargs: original(*args, verbose=verbose, **kwargs)  # pylint: disable=unnecessary-lambda
    try:
        model.fit_generator(
            train_data_loader,
            steps_per_epoch=-(-len(train_data_loader) // tk.hvd.size()),  # ceiling
            validation_data=val_data_loader,
            validation_steps=-(-len(val_data_loader) // tk.hvd.size()) if val_data_loader is not None else None,  # ceiling
            class_weight=class_weight,
            epochs=epochs, callbacks=callbacks,
            verbose=verbose if tk.hvd.is_master() else 0,
            initial_epoch=initial_epoch,
            use_multiprocessing=use_multiprocessing,
            workers=workers, max_queue_size=max_queue_size)
    finally:
        if tf.__version__ == '1.13.1':
            training_generator.model_iteration = original

    # DataLoaderの処理時間を表示
    _logger.info(f'train_data_loader: {train_data_loader.seconds_per_step * 1000:4.0f}ms/step')
    if val_data_loader is not None:
        _logger.info(f'val_data_loader:   {train_data_loader.seconds_per_step * 1000:4.0f}ms/step')


@tk.log.trace()
def predict(model: keras.models.Model, dataset: tk.data.Dataset, batch_size, verbose=1, use_horovod=False):
    """予測。

    Args:
        model: モデル。
        dataset: 予測したい入力データ。
        batch_size (int): バッチサイズ
        verbose (int): プログレスバー(tqdm)を表示するか否か。
        use_horovod (bool): MPIによる分散処理をするか否か。

    Returns:
        np.ndarray: 予測結果。

    """
    dataset = tk.hvd.split(dataset) if use_horovod else dataset
    data_loader = tk.data.DataLoader(dataset, batch_size)
    values = model.predict_generator(data_loader, verbose=verbose if tk.hvd.is_master() else 0)
    values = tk.hvd.allgather(values) if use_horovod else values
    return values


@tk.log.trace()
def evaluate(model: keras.models.Model, dataset, batch_size=32, verbose=1, use_horovod=False):
    """評価。

    Args:
        model: モデル。
        dataset (tk.data.Dataset): データ。
        verbose (int): 1ならプログレスバー表示。
        use_horovod (bool): MPIによる分散処理をするか否か。

    Returns:
        dict: metricsの文字列と値のdict

    """
    dataset = tk.hvd.split(dataset) if use_horovod else dataset
    data_loader = tk.data.DataLoader(dataset, batch_size)
    values = model.evaluate_generator(data_loader, verbose=verbose if tk.hvd.is_master() else 0)
    values = tk.hvd.allreduce(values) if use_horovod else values
    evals = dict(zip(model.metrics_names, values))
    return evals


@tk.log.trace()
def custom_predict(model: keras.models.Model, dataset: tk.data.Dataset, batch_size, verbose=1, desc='predict', on_batch_fn=None, use_horovod=False):
    """予測。

    TTAなど用。

    Args:
        model: モデル。
        dataset (tk.data.Dataset): 予測したい入力データ。
        batch_size (int): バッチサイズ
        verbose (int): プログレスバー(tqdm)を表示するか否か。
        desc (str): プログレスバーの説明。
        on_batch_fn (callable): モデルとミニバッチ分の入力データを受け取り、予測結果を返す処理。(TTA用)
        use_horovod (bool): MPIによる分散処理をするか否か。

    Returns:
        np.ndarray: 予測結果。

    """
    dataset = tk.hvd.split(dataset) if use_horovod else dataset
    verbose = 0 if use_horovod and not tk.hvd.is_master() else verbose
    values = np.array(list(custom_predict_flow(model, dataset, batch_size, verbose, desc, on_batch_fn)))
    values = tk.hvd.allgather(values) if use_horovod else values
    return values


def custom_predict_flow(model: keras.models.Model, dataset: tk.data.Dataset, batch_size, verbose=1, desc='predict', on_batch_fn=None):
    """予測。(yieldバージョン)

    TTAなど用。Horovodでの分散処理機能は無し。(複数GPUで処理したい場合はmulti_gpu_modelを使用するか呼ぶ側でsplitする。処理順に注意が必要。)

    Args:
        model: モデル。
        dataset (tk.data.Dataset): 予測したい入力データ。
        batch_size (int): バッチサイズ
        verbose (int): プログレスバー(tqdm)を表示するか否か。
        desc (str): プログレスバーの説明。
        on_batch_fn (callable): モデルとミニバッチ分の入力データを受け取り、予測結果を返す処理。(TTA用)

    Yields:
        np.ndarray: 予測結果。(サンプル毎)

    """
    if on_batch_fn is None:
        on_batch_fn = _predict_on_batch
    data_loader = tk.data.DataLoader(dataset, batch_size)
    for X, _ in tk.utils.tqdm(data_loader, desc=desc, total=len(data_loader), disable=verbose < 1):
        pred_batch = on_batch_fn(model, X)
        yield from pred_batch


def _predict_on_batch(model: keras.models.Model, X):
    return model.predict_on_batch(X)


@tk.log.trace()
def multi_gpu_model(model, batch_size, gpus=None):
    """複数GPUでデータ並列するモデルを作成する。

    References:
        - <https://github.com/fchollet/keras/issues/2436>
        - <https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py>

    """
    if gpus is None:
        gpus = tk.dl.get_gpu_count()
        tk.log.get(__name__).info(f'gpu count = {gpus}')
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
    parallel_model.save_weights = type(model.save_weights)(_save_weights, parallel_model)

    return parallel_model, batch_size * gpus
