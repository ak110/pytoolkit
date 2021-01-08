"""キャッシュ関連。"""
from __future__ import annotations

import functools
import hashlib
import inspect
import pathlib
import typing

import joblib

import pytoolkit as tk


def memoize(
    cache_dir: typing.Optional[tk.typing.PathLike] = "./cache",
    prefix: str = "",
    compress: int = 0,
):
    """関数の戻り値をファイルにキャッシュするデコレーター。

    Args:
        cache_dir: 保存先ディレクトリ。Noneならキャッシュしない。
        prefix: キャッシュファイル名のプレフィクス (同名関数の衝突を避ける用)
        compress: 0～9の圧縮レベル。一般的には3がおすすめ。

    """

    def decorator(func):
        if cache_dir is None:
            return func

        @functools.wraps(func)
        def memorized_func(*args, **kwargs):
            assert cache_dir is not None
            cache_path = get_cache_path(cache_dir, func, args, kwargs, prefix)
            return memoized_call(lambda: func(*args, **kwargs), cache_path, compress)

        return memorized_func

    return decorator


T = typing.TypeVar("T")


def memoized_call(
    func: typing.Callable[[], T],
    cache_path: typing.Optional[tk.typing.PathLike],
    compress: int = 0,
) -> T:
    """キャッシュがあれば読み、無ければ実処理してキャッシュとして保存する。

    Args:
        func: 実処理
        cache_path: キャッシュの保存先パス。Noneならキャッシュしない。
        compress: 0～9の圧縮レベル。一般的には3がおすすめ。

    """
    if cache_path is None:
        return func()
    cache_path = pathlib.Path(cache_path)
    # キャッシュがあれば読む
    if cache_path.is_file():
        tk.log.get(__name__).info(f"Cache is found: {cache_path}")
        return joblib.load(cache_path)
    else:
        tk.log.get(__name__).info(f"Cache is not found: {cache_path}")
    # 無ければ実処理してキャッシュとして保存
    result = func()
    if tk.hvd.is_master():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, cache_path, compress=compress)
    tk.hvd.barrier()
    if not tk.hvd.is_master():
        result = typing.cast(T, joblib.load(cache_path))
    tk.hvd.barrier()
    return result


def get_cache_path(
    cache_dir: tk.typing.PathLike,
    func: typing.Callable[..., typing.Any],
    args: typing.Sequence[typing.Any],
    kwargs: typing.Dict[str, typing.Any],
    prefix: str = "",
) -> pathlib.Path:
    """キャッシュのパスを作って返す。"""
    cache_dir = pathlib.Path(cache_dir)

    bound_args = inspect.signature(func).bind(*args, **kwargs).arguments
    args_list = sorted(dict(bound_args).items())
    args_str = ",".join([f"{repr(k)}:{repr(v)}" for k, v in args_list])
    cache_hash = hashlib.md5(args_str.encode("utf-8")).hexdigest()[:8]

    tk.log.get(__name__).debug(f"Cache {cache_hash}: arguments={args_str}")
    cache_path = cache_dir / f"{prefix}_{func.__name__}_{cache_hash}.pkl"
    return cache_path
