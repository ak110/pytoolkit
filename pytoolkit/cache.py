"""キャッシュ関連。"""
import functools
import hashlib
import inspect
import pathlib

import sklearn.externals.joblib as joblib


def memorize(cache_dir, compress=0):
    """関数の戻り値をファイルにキャッシュするデコレーター。"""
    cache_dir = pathlib.Path(cache_dir)

    def _decorator(func):
        @functools.wraps(func)
        def _decorated_func(*args, **kwargs):
            cache_path = get_cache_path(cache_dir, func, args, kwargs)
            # キャッシュがあれば読む
            if cache_path.is_file():
                return joblib.load(cache_path)
            # 無ければ実処理
            result = func(*args, **kwargs)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(result, cache_path, compress=compress)
            return result

        return _decorated_func

    return _decorator


def get_cache_path(cache_dir, func, args, kwargs):
    """キャッシュのパスを作って返す。"""
    bound_args = inspect.signature(func).bind(*args, **kwargs).arguments
    args_list = sorted(dict(bound_args).items())
    args_str = ','.join([f'{repr(k)}:{repr(v)}' for k, v in args_list])
    args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()[:8]
    cache_path = cache_dir / f'{func.__name__}_{args_hash}.pkl'
    return cache_path
