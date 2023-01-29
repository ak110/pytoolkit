"""ログ関連。

fmtに色々出したいときはこの辺を参照：
https://docs.python.jp/3/library/logging.html#logrecord-attributes

"""
import functools
import logging
import logging.handlers
import pathlib
import time

import tensorflow as tf

import pytoolkit as tk


def init(
    output_path,
    append=False,
    rotate=False,
    max_bytes=1048576,
    backup_count=10,
    stream_level=logging.INFO,
    stream_fmt="[%(levelname)-5s] %(message)s",
    file_level=logging.DEBUG,
    file_fmt="%(asctime)s [%(levelname)-5s] %(message)s <%(name)s> %(filename)s:%(lineno)d",
    matplotlib_level=logging.WARNING,
    pil_level=logging.INFO,
    close_tf_logger=True,
    use_lightgbm: bool = False,
):
    """ルートロガーの初期化。"""
    logger = get(None)
    logger.setLevel(logging.DEBUG)
    close(logger)
    logger.addHandler(stream_handler(level=stream_level, fmt=stream_fmt))
    if output_path is not None and tk.hvd.is_master():
        logger.addHandler(
            file_handler(
                output_path,
                append,
                rotate,
                max_bytes,
                backup_count,
                level=file_level,
                fmt=file_fmt,
            )
        )
    # ログが出すぎる他のライブラリなどのための調整
    if matplotlib_level is not None:
        get("matplotlib").setLevel(matplotlib_level)
    if pil_level is not None:
        get("PIL").setLevel(pil_level)
    if close_tf_logger:
        assert tf.get_logger().propagate
        tk.log.close(tf.get_logger())
    if use_lightgbm:
        import lightgbm as lgb

        lgb.register_logger(get("lightgbm"))


def get(name):
    """ロガーを取得して返す。"""
    return logging.getLogger(name)


def stream_handler(
    stream=None, level=logging.INFO, fmt="[%(levelname)-5s] %(message)s"
):
    """StreamHandlerを作成して返す。levelは文字列で'DEBUG'とかも指定可。(Python>=3.2)"""
    handler = logging.StreamHandler(stream=stream)
    handler.setLevel(level)
    if fmt:
        handler.setFormatter(logging.Formatter(fmt))
    return handler


def file_handler(
    output_path,
    append=False,
    rotate=False,
    max_bytes=1048576,
    backup_count=10,
    encoding="utf-8",
    level=logging.DEBUG,
    fmt="[%(levelname)-5s] %(message)s <%(name)s:%(filename)s:%(lineno)d>",
):
    """RotatingFileHandler/FileHandlerを作成して返す。levelは文字列で'INFO'とかも指定可。"""
    output_path = pathlib.Path(output_path)
    output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    if rotate:
        handler: logging.Handler = logging.handlers.RotatingFileHandler(
            str(output_path), "a", max_bytes, backup_count, encoding=encoding
        )
    else:
        handler = logging.FileHandler(
            str(output_path), "a" if append else "w", encoding=encoding
        )
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


class trace:
    """開始・終了をログるdecorator＆context manager。

    Args:
        process_name: ログに出力する処理の名前。(Noneなら関数名。context managerとしての使用時は必須)
        level: ログレベル。文字列で'DEBUG'とかも指定可。

    Examples:
        decoratorの例::

            @tk.log.trace()
            def func()
                pass

        context managerの例::

            with tk.log.trace("process"):
                process()

    """

    def __init__(self, process_name=None, level=logging.INFO):
        self.process_name = process_name
        self.level = level
        self.start_time = None

    def __call__(self, func):
        @functools.wraps(func)
        def traced_func(*args, **kwds):
            with self.__class__(
                process_name=self.process_name or func.__qualname__, level=self.level
            ):
                return func(*args, **kwds)

        return traced_func

    def __enter__(self):
        assert self.process_name is not None
        assert self.start_time is None
        get(__name__).log(self.level, f"{self.process_name} start")
        self.start_time = time.time()
        return self

    def __exit__(self, *exc):
        assert self.start_time is not None
        elapsed_time = time.time() - self.start_time
        if exc[0] is None:
            get(__name__).log(
                self.level, f"{self.process_name} done in {elapsed_time:.1f} s"
            )
        else:
            get(__name__).log(
                self.level, f"{self.process_name} error in {elapsed_time:.1f} s"
            )
        self.start_time = None  # チェック用に戻しておく
        return False
