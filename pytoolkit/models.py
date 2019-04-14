"""Kerasのモデル関連。"""
import os
import pathlib
import sys

import numpy as np

from . import callbacks as cb, data, hvd, keras, log, utils


def load(path: pathlib.Path, custom_objects=None, compile=False) -> keras.models.Model:  # pylint: disable=redefined-outer-name
    """モデルの読み込み。"""
    from . import get_custom_objects
    custom_objects = custom_objects.copy() if custom_objects else dict()
    custom_objects.update(get_custom_objects())
    return keras.models.load_model(str(path), custom_objects=custom_objects, compile=compile)


def load_weights(model: keras.models.Model, path: pathlib.Path, by_name=False):
    """存在すればload_weights。"""
    path = pathlib.Path(path)
    if path.exists():
        with log.trace_scope(f'load_weights({path.name})'):
            model.load_weights(str(path), by_name=by_name)
    else:
        log.get(__name__).info(f'{path.name} is not found.')


@log.trace()
def save(model: keras.models.Model, path: pathlib.Path, include_optimizer=False):
    """モデルの保存。"""
    if hvd.is_master():
        model.save(str(path), include_optimizer=include_optimizer)
    hvd.barrier()


def summary(model: keras.models.Model):
    """summaryを実行するだけ。"""
    model.summary(print_fn=log.get(__name__).info if hvd.is_master() else lambda x: None)


def compile(model: keras.models.Model, optimizer, loss, metrics=None, loss_weights=None):  # pylint: disable=redefined-builtin
    """compileするだけ。"""
    if hvd.initialized():
        optimizer = hvd.get().DistributedOptimizer(optimizer, compression=hvd.get().Compression.fp16)
    model.compile(optimizer, loss, metrics, loss_weights=loss_weights)


def fit(model: keras.models.Model, training_data: data.Dataset, validation_data: data.Dataset = None,
        batch_size=32, epochs=1800,
        callbacks=None, verbose=1,
        mixup=False):
    """独自のtraining loopになる予定の関数。

    Args:
        model: モデル。
        training_data (tk.data.Dataset): 訓練データ。
        validation_data (tk.data.Dataset): 検証データ。Noneなら省略。
        callbacks (list): コールバック。EpochLoggerとErrorOnNaNは自動追加。
        verbose (int): 1ならプログレスバー表示、2ならepoch毎の結果だけ表示。
        mixup (bool): mixupするのか否か。

    """
    callbacks = (callbacks or []) + [
        cb.EpochLogger(),
        cb.ErrorOnNaN(),
    ]
    mp_size = hvd.get().size()
    train_data = data.DataLoader(training_data, batch_size, shuffle=True, mixup=mixup, mp_size=mp_size)
    val_data = data.DataLoader(validation_data, batch_size * 2, shuffle=True, mp_size=mp_size) if validation_data is not None else None

    if not hvd.is_master():
        # TensorFlow 1.13.1くらい用のworkaround
        backup_stdout = sys.stdout
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
    try:
        model.fit_generator(
            train_data, validation_data=val_data,
            epochs=epochs, callbacks=callbacks,
            verbose=verbose if hvd.is_master() else 0)
    finally:
        if not hvd.is_master():
            devnull.close()
            sys.stdout = backup_stdout


def predict(model: keras.models.Model, predict_data: data.Dataset, batch_size, verbose=1, desc='predict', on_batch_fn=None):
    """予測。

    Args:
        model: モデル。
        predict_data: 予測したい入力データ。
        batch_size (int): バッチサイズ
        verbose (int): プログレスバー(tqdm)を表示するか否か。
        desc (str): プログレスバーの説明。
        on_batch_fn (callable): モデルとミニバッチ分の入力データを受け取り、予測結果を返す処理。(TTA用)

    Returns:
        np.ndarray: 予測結果。

    """
    return np.array(list(predict_flow(model, predict_data, batch_size, verbose, desc, on_batch_fn)))


def predict_flow(model: keras.models.Model, predict_data: data.Dataset, batch_size, verbose=1, desc='predict', on_batch_fn=None):
    """予測。(yieldバージョン)

    Args:
        model: モデル。
        predict_data: 予測したい入力データ。
        batch_size (int): バッチサイズ
        verbose (int): プログレスバー(tqdm)を表示するか否か。
        desc (str): プログレスバーの説明。
        on_batch_fn (callable): モデルとミニバッチ分の入力データを受け取り、予測結果を返す処理。(TTA用)

    Yields:
        np.ndarray: 予測結果。(サンプル毎)

    """
    if on_batch_fn is None:
        on_batch_fn = _predict_on_batch
    data_loader = data.DataLoader(predict_data, batch_size)
    for X, _ in utils.tqdm(data_loader, desc=desc, total=len(data_loader), disable=verbose < 1):
        pred_batch = on_batch_fn(model, X)
        yield from pred_batch


def _predict_on_batch(model: keras.models.Model, X):
    return model.predict_on_batch(X)
