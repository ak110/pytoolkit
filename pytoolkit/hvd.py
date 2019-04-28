"""Horovodの薄いwrapper。"""
import logging
import pathlib

import tensorflow as tf
from . import keras

_logger = logging.getLogger(__name__)
_initialized = False


def get():
    """horovod.kerasモジュールを返す。"""
    if keras == tf.keras:
        import horovod.tensorflow.keras as hvd
    else:
        import horovod.keras as hvd
    return hvd


def init():
    """初期化。"""
    global _initialized
    if not _initialized:
        try:
            get().init()
            _initialized = True
        except ImportError:
            _logger.warning('Horovod読み込み失敗', exc_info=True)


def initialized():
    """初期化済みなのか否か(Horovodを使うのか否か)"""
    return _initialized


def is_master():
    """Horovod未使用 or hvd.rank() == 0ならTrue。"""
    if not initialized():
        return True
    return get().rank() == 0


def is_local_master():
    """Horovod未使用 or hvd.local_rank() == 0ならTrue。"""
    if not initialized():
        return True
    return get().local_rank() == 0


def barrier():
    """全員が揃うまで待つ。"""
    if not initialized():
        return
    import mpi4py
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm.barrier()


def bcast(buf, root=0):
    """MPI_Bcastを呼び出す。"""
    if not initialized():
        return buf
    import mpi4py
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    return comm.bcast(buf, root)


def get_file(name, url, **kwargs):
    """local_masterだけkeras.utils.get_fileを呼び出す。"""
    if is_local_master():
        keras.utils.get_file(name, url, **kwargs)
    barrier()
    return pathlib.Path(keras.utils.get_file(name, url, **kwargs))
