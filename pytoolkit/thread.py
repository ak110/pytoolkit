"""スレッド周りの処理。"""
import atexit
import logging
import threading


def start_background_loop(fn, args=None, kwargs=None, interval=1):
    """別スレッドでループして定期的にfnを呼び出す。

    # 引数
    - fn: callable
    - interval: fnを呼び出す間隔

    """
    args = args or []
    kwargs = kwargs or {}
    timer = None

    def _run():
        try:
            fn(*args, **kwargs)
        except BaseException:
            logger = logging.getLogger(__name__)
            logger.warning('background loopで例外発生', exc_info=True)
        timer = threading.Timer(interval, _run)
        timer.start()

    def _stop():
        timer.cancel()

    timer = threading.Timer(interval, _run)
    timer.start()
    atexit.register(_stop)


def start_thread(fn, args=None, kwargs=None):
    """作りっぱなしなスレッド。"""
    args = args or []
    kwargs = kwargs or {}
    thread = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=False)
    thread.start()
    return thread
