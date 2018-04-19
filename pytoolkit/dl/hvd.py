"""Horovodの薄いwrapper。"""

_initialized = False


def get():
    """`horovod.keras`モジュールを返す。"""
    import horovod.keras as hvd
    return hvd


def init():
    """初期化。"""
    global _initialized
    _initialized = True
    get().init()


def initialized():
    """初期化済みなのか否か(Horovodを使うのか否か)"""
    return _initialized


def is_master():
    """Horovod未使用 or hvd.rank() == 0ならTrue。"""
    if not initialized():
        return True
    get().init()
