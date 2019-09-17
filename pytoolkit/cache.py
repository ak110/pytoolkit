"""キャッシュ関連。"""
import functools
import hashlib
import inspect
import pathlib

import joblib

import pytoolkit as tk


def memorize(cache_dir, compress=0, verbose=True):
    """関数の戻り値をファイルにキャッシュするデコレーター。

    force_rerun=Trueを付けて呼び出すと強制的に再実行してキャッシュを上書き。

    """
    cache_dir = pathlib.Path(cache_dir)

    def _decorator(func):
        @functools.wraps(func)
        def _decorated_func(*args, force_rerun=False, **kwargs):
            cache_path = get_cache_path(cache_dir, func, args, kwargs)
            # キャッシュがあれば読む
            if not force_rerun:
                if cache_path.is_file():
                    if verbose:
                        tk.log.get(__name__).info(f"Cache is found: {cache_path}")
                    return joblib.load(cache_path)
                else:
                    if verbose:
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

        return _decorated_func

    return _decorator


def get_cache_path(cache_dir, func, args, kwargs):
    """キャッシュのパスを作って返す。"""
    bound_args = inspect.signature(func).bind(*args, **kwargs).arguments
    args_list = sorted(dict(bound_args).items())
    args_str = ",".join([f"{repr(k)}:{repr(v)}" for k, v in args_list])
    args_hash = hashlib.md5(args_str.encode("utf-8")).hexdigest()[:8]
    cache_path = cache_dir / f"{func.__name__}_{args_hash}.pkl"
    return cache_path
