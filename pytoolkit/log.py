"""ログ関連。

fmtに色々出したいときはこの辺を参照：
https://docs.python.jp/3/library/logging.html#logrecord-attributes

"""
import contextlib
import functools
import logging
import logging.handlers
import pathlib
import time


def init(output_path, append=False, rotate=False, max_bytes=1048576, backup_count=10,
         stream_level=logging.INFO,
         stream_fmt='[%(levelname)-5s] %(message)s',
         file_level=logging.DEBUG,
         file_fmt='[%(levelname)-5s] %(message)s <%(name)s:%(filename)s:%(lineno)d>',
         matplotlib_level=logging.WARNING,
         pil_level=logging.INFO):
    """ルートロガーの初期化。"""
    from .dl import hvd
    logger = get(None)
    logger.setLevel(logging.DEBUG)
    close(logger)
    logger.addHandler(stream_handler(level=stream_level, fmt=stream_fmt))
    if output_path is not None and hvd.is_master():
        logger.addHandler(file_handler(output_path, append, rotate, max_bytes, backup_count,
                                       level=file_level, fmt=file_fmt))
    if matplotlib_level is not None:
        get('matplotlib').setLevel(matplotlib_level)
    if pil_level is not None:
        get('PIL').setLevel(pil_level)


def get(name):
    """ロガーを取得して返す。"""
    return logging.getLogger(name)


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
                 fmt='[%(levelname)-5s] %(message)s <%(name)s:%(filename)s:%(lineno)d>'):
    """RotatingFileHandler/FileHandlerを作成して返す。levelは文字列で'INFO'とかも指定可。(Python>=3.2)"""
    output_path = pathlib.Path(output_path)
    output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
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

    Args:
        process_name: ログに出力する処理の名前。(Noneなら関数名)
        logger_name: ロガーの名前。

    """
    def _decorator(func):
        @functools.wraps(func)
        def _decorated_func(*args, **kwargs):
            with trace_scope(process_name or func.__qualname__, logger_name):
                return func(*args, **kwargs)
        return _decorated_func
    return _decorator


@contextlib.contextmanager
def trace_scope(process_name, logger_name=__name__):
    """withで使うと、処理前後でログを出力する。

    Args:
        process_name: ログに出力する処理の名前。
        logger_name: ロガーの名前。

    """
    logger = get(logger_name)
    logger.debug('%s 開始', process_name)
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        logger.debug('%s 終了 (%.3f[s])', process_name, elapsed_time)
