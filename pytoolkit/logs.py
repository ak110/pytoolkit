"""ログ関連のヘルパー"""
import contextlib
import logging
import os
import pathlib
import time


def init(
    output_path: str | os.PathLike[str] | None = None,
    output_mode: str = "w",
    stream_level=logging.INFO,
    stream_fmt="[%(levelname)-5s] %(message)s",
    file_level=logging.DEBUG,
    file_fmt="[%(levelname)-5s] %(message)s <%(name)s> %(filename)s:%(lineno)d",
) -> None:
    """初期化"""
    handlers: list[logging.Handler] = []
    handlers.append(logging.StreamHandler())
    handlers[-1].setLevel(stream_level)
    handlers[-1].setFormatter(logging.Formatter(stream_fmt))
    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output_path, output_mode))
        handlers[-1].setLevel(file_level)
        handlers[-1].setFormatter(logging.Formatter(file_fmt))
    logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)
    get("matplotlib").setLevel(logging.WARNING)
    get("PIL").setLevel(logging.INFO)


def get(name: str) -> logging.Logger:
    """ロガーの取得"""
    return logging.getLogger(name)


@contextlib.contextmanager
def timer(name: str, logger: logging.Logger | None = None):
    """処理時間の計測＆表示。"""
    start_time = time.perf_counter()
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info(f"<{name}> start.")
    yield
    logger.info(f"<{name}> done in {time.perf_counter() - start_time:.0f} s.")
