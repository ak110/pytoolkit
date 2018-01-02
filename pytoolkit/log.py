"""ログ関連"""
import contextlib
import functools
import logging
import logging.handlers
import time


def get(name=None):
    """"ロガーを取得して返す。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger


def stream_handler(stream=None, level=logging.INFO, fmt='[%(levelname)-5s] %(message)s'):
    """StreamHandlerを作成して返す。levelは文字列で'DEBUG'とかも指定可。(Python>=3.2)"""
    handler = logging.StreamHandler(stream=stream)
    handler.setLevel(level)
    if fmt:
        handler.setFormatter(logging.Formatter(fmt))
    return handler


def file_handler(output_path, append=False, rotate=False,
                 max_bytes=1048576, backup_count=10, encoding='utf-8',
                 level=logging.DEBUG,
                 fmt='%(asctime)s [%(levelname)-5s] [%(filename)s:%(lineno)d] %(message)s'):
    """RotatingFileHandler / FileHandlerを作成して返す。levelは文字列で'INFO'とかも指定可。(Python>=3.2)"""
    if rotate:
        handler = logging.handlers.RotatingFileHandler(
            str(output_path), 'a', max_bytes, backup_count, encoding=encoding)
    else:
        handler = logging.FileHandler(str(output_path), 'a' if append else 'w', encoding=encoding)
    handler.setLevel(level)
    if fmt:
        handler.setFormatter(logging.Formatter(fmt))
    return handler


def close(logger):
    """loggerが持っているhandlerを全部closeしてremoveする。"""
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def trace(process_name=None, logger_name=__name__):
    """関数の開始・終了をログるdecorator。

    # 引数
    - process_name: ログに出力する処理の名前。(Noneなら関数名)
    - logger_name: ロガーの名前。

    """
    def _decorator(func):
        @functools.wraps(func)
        def _decorated_func(*args, **kwargs):
            with trace_scope(process_name or func.__name__, logger_name):
                return func(*args, **kwargs)
        return _decorated_func
    return _decorator


@contextlib.contextmanager
def trace_scope(process_name, logger_name=__name__):
    """withで使うと、処理前後でログを出力する。

    # 引数
    - process_name: ログに出力する処理の名前。
    - logger_name: ロガーの名前。

    """
    logger = get(logger_name)
    logger.debug('%s 開始', process_name)
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        logger.debug('%s 終了 (%.3f[s])', process_name, elapsed_time)
