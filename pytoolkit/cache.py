"""キャッシュ関連。"""
from __future__ import annotations

import functools
import hashlib
import inspect
import pathlib
import typing

import joblib

import pytoolkit as tk


def memorize(cache_dir: tk.typing.PathLike = "./cache", compress: int = 0):
    """関数の戻り値をファイルにキャッシュするデコレーター。

    Args:
        cache_dir: 保存先ディレクトリ
        compress: 0～9の圧縮レベル。一般的には3がおすすめ。

    """

    def decorator(func):
        @functools.wraps(func)
        def memorized_func(*args, **kwargs):
            cache_path = get_cache_path(cache_dir, func, args, kwargs)
            # キャッシュがあれば読む
            if cache_path.is_file():
                tk.log.get(__name__).info(f"Cache is found: {cache_path}")
                return joblib.load(cache_path)
            else:
                tk.log.get(__name__).info(f"Cache is not found: {cache_path}")
            # 無ければ実処理
            result = func(*args, **kwargs)
            if tk.hvd.is_master():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(result, cache_path, compress=compress)
            tk.hvd.barrier()
            if not tk.hvd.is_master():
                result = joblib.load(cache_path)
            tk.hvd.barrier()
            return result

        return memorized_func

    return decorator


def get_cache_path(
    cache_dir: tk.typing.PathLike,
    func: typing.Callable,
    args: typing.Sequence,
    kwargs: typing.Dict,
):
    """キャッシュのパスを作って返す。"""
    cache_dir = pathlib.Path(cache_dir)
    bound_args = inspect.signature(func).bind(*args, **kwargs).arguments
    args_list = sorted(dict(bound_args).items())
    args_str = ",".join([f"{repr(k)}:{repr(v)}" for k, v in args_list])
    args_hash = hashlib.md5(args_str.encode("utf-8")).hexdigest()[:8]
    tk.log.get(__name__).info(f"Cache {args_hash}: arguments={args_str}")
    cache_path = cache_dir / f"{func.__name__}_{args_hash}.pkl"
    return cache_path
