"""multiple input/outputに対応するためのヘルパー関数たち。

listならmultipleと見なす。
"""


def is_none(arr):
    """Noneなのか否か判定。"""
    if arr is None:
        return True
    if isinstance(arr, list):
        assert all([(arr[0] is None) == (a is None) for a in arr[1:]])
        return arr[0] is None
    return False


def get_length(arr):
    """長さを返す。"""
    if isinstance(arr, list):
        assert all([len(arr[0]) == len(a) for a in arr[1:]])
        return len(arr[0])
    return len(arr)


def get(arr, ix):
    """指定インデックスの要素を返す。"""
    if arr is None:
        return None
    if isinstance(arr, list):
        return [x[ix] for x in arr]
    return arr[ix]
