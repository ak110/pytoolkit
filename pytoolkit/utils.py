"""各種ユーティリティ"""
import pathlib
import os
import joblib

import pytoolkit as tk


def find_by_name(arr, name):
    """__name__から要素を検索して返す。"""
    for x in arr:
        if x.__name__ == name:
            return x
    raise ValueError(f'"{name}" is not exist in [{arr}]')


def normalize_tuple(value, n):
    """n個の要素を持つtupleにして返す。ただしNoneならNoneのまま。"""
    if value is None:
        return None
    elif isinstance(value, int):
        return (value,) * n
    else:
        value = tuple(value)
        assert len(value) == n
        return value


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
    from tqdm import tqdm as t

    return t(iterable, desc, total, leave, ascii=True, ncols=100, **kwargs)


def trange(*args, **kwargs):
    """ascii=Trueでncols=100なtqdm.trange。"""
    return tqdm(list(range(*args)), **kwargs)


def tqdm_write(s, file=None, end="\n", nolock=False):
    """tqdm中に何か出力したいとき用のやつ。"""
    from tqdm import tqdm as t

    t.write(s, file=file, end=end, nolock=nolock)


def better_exceptions():
    """better_exceptionsを有効にする。"""
    try:
        # サブプロセスとか用に環境変数を設定
        os.environ["BETTER_EXCEPTIONS"] = "1"
        # 今のスレッドでアクティブにする
        import better_exceptions as be

        be.hook()
    except BaseException:
        tk.log.get(__name__).warning("better_exceptions error", exc_info=True)
