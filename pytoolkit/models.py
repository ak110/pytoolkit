"""Kerasのモデル関連。"""
import os
import pathlib
import sys

from . import callbacks as cb, data, hvd, keras, log


def load_weights(model: keras.models.Model, path: pathlib.Path, by_name=False):
    """存在すればload_weights。"""
    path = pathlib.Path(path)
    if path.exists():
        with log.trace_scope(f'load_weights({path.name})'):
            model.load_weights(str(path), by_name=by_name)
    else:
        log.get(__name__).info(f'{path.name} is not found.')


def summary(model: keras.models.Model):
    """summaryを実行するだけ。"""
    model.summary(print_fn=log.get(__name__).info if hvd.is_master() else lambda x: None)


def compile(model: keras.models.Model, optimizer, loss, metrics=None, loss_weights=None):
    """compileするだけ。"""
    if hvd.initialized():
        optimizer = hvd.get().DistributedOptimizer(optimizer, compression=hvd.get().Compression.fp16)
    model.compile(optimizer, loss, metrics, loss_weights=loss_weights)


def fit(model: keras.models.Model, training_data, validation_data=None,
        batch_size=32, epochs=1800,
        callbacks=None, verbose=1,
        mixup=False):
    """独自のtraining loopになる予定の関数。

    Args:
        model: モデル。
        training_data (tk.utils.Dataset): 訓練データ。
        validation_data (tk.utils.Dataset): 検証データ。Noneなら省略。
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


@log.trace()
def save(model: keras.models.Model, path: pathlib.Path, include_optimizer=False):
    """saveを実行するだけ。"""
    if hvd.is_master():
        model.save(str(path), include_optimizer=include_optimizer)
    hvd.barrier()
