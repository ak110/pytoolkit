"""各種ユーティリティ"""
import logging
import os

_logger = logging.getLogger(__name__)


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


def tqdm(iterable=None, desc=None, total=None, leave=True, **kwargs):
    """ascii=Trueでncols=100なtqdm。"""
    from tqdm import tqdm as t
    return t(iterable, desc, total, leave, ascii=True, ncols=100, **kwargs)


def trange(*args, **kwargs):
    """ascii=Trueでncols=100なtqdm.trange。"""
    return tqdm(list(range(*args)), **kwargs)


def better_exceptions():
    """better_exceptionsを有効にする。"""
    try:
        # サブプロセスとか用に環境変数を設定
        os.environ['BETTER_EXCEPTIONS'] = '1'
        # 今のスレッドでアクティブにする
        import better_exceptions as be
        be.hook()
    except BaseException:
        _logger.warning('better_exceptions error', exc_info=True)
