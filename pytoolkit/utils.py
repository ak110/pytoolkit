"""各種ユーティリティ"""
import contextlib
import datetime
import functools
import logging
import os
import pathlib
import pickle
import sys
import traceback
import typing
import warnings

import joblib
import numpy as np

logger = logging.getLogger(__name__)


def find_by_name(arr, name):
    """__name__から要素を検索して返す。"""
    for x in arr:
        if x.__name__ == name:
            return x
    raise ValueError(f'"{name}" is not exist in [{arr}]')


def normalize_tuple(value, n: int) -> typing.Tuple[int, ...]:
    """n個の要素を持つtupleにして返す。"""
    assert value is not None
    if isinstance(value, int):
        return (value,) * n
    else:
        value = tuple(value)
        assert len(value) == n
        return value


def memoize(func):
    """単純なメモ化のデコレーター。"""
    cache: typing.Dict[bytes, typing.Any] = {}

    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = pickle.dumps((args, kwargs))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized_func


def subprocess(func):
    """子プロセスで実行するデコレーター。GPUメモリ周りの対処とか用。"""

    @functools.wraps(func)
    def subprocess_func(*args, **kwargs):
        with joblib.Parallel(n_jobs=2, backend="loky") as parallel:
            return parallel([joblib.delayed(func)(*args, **kwargs)])[0]

    return subprocess_func


def dump(value, filename, compress=0, protocol=None, cache_size=None):
    """ディレクトリを自動的に作るjoblib.dump。"""
    filename = pathlib.Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        value=value,
        filename=filename,
        compress=compress,
        protocol=protocol,
        cache_size=cache_size,
    )


def load(filename, mmap_mode=None, skip_not_exist=False):
    """joblib.loadほぼそのまま。"""
    filename = pathlib.Path(filename)  # 一応dumpに合わせて。
    if skip_not_exist and not filename.exists():
        return None
    return joblib.load(filename, mmap_mode=mmap_mode)


def tqdm(iterable=None, desc=None, total=None, leave=True, **kwargs):
    """ascii=Trueでncols=100なtqdm。"""
    try:
        import tqdm.auto as t

        return t.tqdm(iterable, desc, total, leave, ascii=True, ncols=100, **kwargs)
    except ImportError:
        warnings.warn("tqdm import error")
        return iterable


def trange(*args, **kwargs):
    """ascii=Trueでncols=100なtqdm.trange。"""
    return tqdm(list(range(*args)), **kwargs)


def tenumerate(iterable, **tqdm_kwargs):
    """ascii=Trueでncols=100なtqdm.contrib.tenumerate。"""
    try:
        import tqdm.contrib as t

        return t.tenumerate(iterable, ascii=True, ncols=100, **tqdm_kwargs)
    except ImportError:
        warnings.warn("tqdm import error")
        return enumerate(iterable)


def tzip(iter1, *iter2plus, **tqdm_kwargs):
    """ascii=Trueでncols=100なtqdm.contrib.tzip。"""
    try:
        import tqdm.contrib as t

        return t.tzip(iter1, *iter2plus, ascii=True, ncols=100, **tqdm_kwargs)
    except ImportError:
        warnings.warn("tqdm import error")
        return zip(iter1, *iter2plus)


def tmap(function, *sequences, **tqdm_kwargs):
    """ascii=Trueでncols=100なtqdm.contrib.tmap。"""
    try:
        import tqdm.contrib as t

        return t.tmap(function, *sequences, ascii=True, ncols=100, **tqdm_kwargs)
    except ImportError:
        warnings.warn("tqdm import error")
        return map(function, *sequences)


def tqdm_write(s, file=None, end="\n", nolock=False):
    """tqdm中に何か出力したいとき用のやつ。"""
    try:
        import tqdm as t

        t.tqdm.write(s, file=file, end=end, nolock=nolock)
    except ImportError:
        pass


def tqdm_external_write_mode(file=None, nolock=False):
    """tqdm中に何か出力したいとき用のやつ。"""
    try:
        import tqdm as t

        return t.tqdm.external_write_mode(file=file, nolock=nolock)
    except ImportError:
        return contextlib.nullcontext()


def better_exceptions():
    """better_exceptionsを有効にする。"""
    try:
        # サブプロセスとか用に環境変数を設定
        os.environ["BETTER_EXCEPTIONS"] = "1"
        # 今のスレッドでアクティブにする
        import better_exceptions as be

        be.hook()
    except Exception:
        logger.warning("better_exceptions error", exc_info=True)


def format_exc(color=False, safe=True) -> str:
    """例外をbetter_exceptionsで整形して返す。"""
    exc, value, tb = sys.exc_info()
    return format_exception(exc, value, tb, color=color, safe=safe)


def format_exception(exc, value, tb, color=False, safe=True) -> str:
    """例外をbetter_exceptionsで整形して返す。"""
    try:
        import better_exceptions as be

        formatter = be.ExceptionFormatter(
            colored=color and be.SUPPORTS_COLOR,
            theme=be.THEME,
            max_length=be.MAX_LENGTH,
            pipe_char=be.PIPE_CHAR,
            cap_char=be.CAP_CHAR,
        )
        return formatter.format_exception(exc, value, tb)

    except Exception:
        if not safe:
            raise
        return "".join(traceback.format_exception(exc, value, tb))


def encode_rl_array(masks, desc="encode_rl") -> typing.List[str]:
    """encode_rlの配列版。"""
    return [encode_rl(m) for m in tqdm(masks, desc=desc)]


def encode_rl(mask: np.ndarray) -> str:
    """Kaggleのセグメンテーションで使われるようなランレングスエンコード。

    Args:
        mask (ndarray): 0 or 1のマスク画像 (shapeは(height, width)または(height, width, 1))

    Returns:
        エンコードされた文字列

    """
    mask = np.squeeze(mask)
    assert mask.ndim == 2
    mask = mask.reshape(mask.shape[0] * mask.shape[1], order="F")
    mask = np.concatenate([[0], mask, [0]])  # 番兵
    changes = mask[1:] != mask[:-1]  # 隣と比較
    rls = np.where(changes)[0] + 1  # 隣と異なる位置のindex
    rls[1::2] -= rls[::2]  # 開始位置＋サイズ
    return " ".join(str(x) for x in rls)


def daterange(
    start: datetime.date, end: datetime.date
) -> typing.Iterator[datetime.date]:
    """startからend-1の日付を列挙"""
    for i in range((end - start).days):
        yield start + datetime.timedelta(i)
