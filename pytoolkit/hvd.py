"""Horovodの薄いwrapper。"""
from __future__ import annotations

import tensorflow as tf

import pytoolkit as tk

_initialized = False


def get():
    """horovod.tf.kerasモジュールを返す。"""
    import horovod.tensorflow.keras as hvd

    return hvd


def init() -> None:
    """初期化。"""
    global _initialized
    if not _initialized:
        try:
            get().init()

            # mpi4pyを使うにはmulti-threadingが必要らしい
            # https://github.com/horovod/horovod#mpi4py
            assert get().mpi_threads_supported()

            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.experimental.set_visible_devices(
                    gpus[get().local_rank()], "GPU"
                )
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
    if initialized():
        value = get().allgather(value)
        # tensorが来たらnumpy化
        if hasattr(value, "numpy"):
            value = value.numpy()
    return value


def allreduce(value, average: bool = True):
    """全ワーカーからデータを集める。"""
    if initialized():
        value = get().allreduce(value, average=average)
        # tensorが来たらnumpy化
        if hasattr(value, "numpy"):
            value = value.numpy()
    return value


def barrier() -> None:
    """全員が揃うまで待つ。"""
    if not initialized():
        return
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    comm.barrier()


def bcast(buf, root=0):
    """MPI_Bcastを呼び出す。"""
    if not initialized():
        return buf
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    return comm.bcast(buf, root)


def get_file(name, url, **kwargs) -> str:
    """local_masterだけtf.keras.utils.get_fileを呼び出す。"""
    if is_local_master():
        tf.keras.utils.get_file(name, url, **kwargs)
    barrier()
    return tf.keras.utils.get_file(name, url, **kwargs)


def split(dataset: tk.data.Dataset) -> tk.data.Dataset:
    """Datasetを各ワーカーで分割処理する。

    処理結果は hvd.allgather() や hvd.allreduce() で集めることができる。

    Args:
        dataset: 分割元のデータセット

    Returns:
        分割後のデータセット

    """
    if initialized():
        dataset = tk.data.split(dataset, get().size())[get().rank()]
    return dataset
