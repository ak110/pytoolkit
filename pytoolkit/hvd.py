"""Horovodの薄いwrapper。"""
from __future__ import annotations

import tensorflow as tf

import pytoolkit as tk

from . import keras

_initialized = False


def get():
    """horovod.kerasモジュールを返す。"""
    if keras == tf.keras:
        import horovod.tensorflow.keras as hvd
    else:
        import horovod.keras as hvd
    return hvd


def init() -> None:
    """初期化。"""
    global _initialized
    if not _initialized:
        try:
            get().init()
            _initialized = True
        except ImportError:
            tk.log.get(__name__).warning("Horovod読み込み失敗", exc_info=True)


def initialized() -> bool:
    """初期化済みなのか否か(Horovodを使うのか否か)"""
    return _initialized


def size() -> int:
    """hvd.size。"""
    return get().size() if initialized() else 1


def rank() -> int:
    """hvd.rank。"""
    return get().rank() if initialized() else 0


def local_rank() -> int:
    """hvd.local_rank。"""
    return get().local_rank() if initialized() else 0


def is_master() -> bool:
    """Horovod未使用 or hvd.rank() == 0ならTrue。"""
    return rank() == 0


def is_local_master() -> bool:
    """Horovod未使用 or hvd.local_rank() == 0ならTrue。"""
    return local_rank() == 0


def allgather(value):
    """全ワーカーからデータを集める。"""
    return get().allgather(value) if initialized() else value


def allreduce(value, average: bool = True):
    """全ワーカーからデータを集める。"""
    return get().allreduce(value, average=average) if initialized() else value


def barrier() -> None:
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


def get_file(name, url, **kwargs) -> str:
    """local_masterだけkeras.utils.get_fileを呼び出す。"""
    if is_local_master():
        keras.utils.get_file(name, url, **kwargs)
    barrier()
    return keras.utils.get_file(name, url, **kwargs)


def split(dataset: tk.data.Dataset) -> tk.data.Dataset:
    """Datasetを各ワーカーで分割処理する。

    処理結果は hvd.allgather() や hvd.allreduce() で集めることができる。

    Args:
        dataset: 分割元のデータセット

    Returns:
        分割後のデータセット

    """
    from . import data

    if not initialized():
        return dataset
    return data.split(dataset, get().size())[get().rank()]
