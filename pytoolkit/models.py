"""Kerasのモデル関連。"""
import pathlib

from . import hvd, keras, log


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


@log.trace()
def save(model: keras.models.Model, path: pathlib.Path, include_optimizer=False):
    """saveを実行するだけ。"""
    if hvd.is_master():
        model.save(str(path), include_optimizer=include_optimizer)
    hvd.barrier()
