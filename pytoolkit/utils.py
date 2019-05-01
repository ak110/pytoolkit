"""各種ユーティリティ"""
import functools
import logging
import os
import subprocess

import numpy as np

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


def memoize(func):
    """単純なメモ化のデコレーター。"""
    cache = {}

    @functools.wraps(func)
    def _helper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return _helper


def nvidia_smi(*args):
    """nvidia-smiコマンドを実行する。"""
    path = os.path.join(os.environ.get('ProgramFiles', ''), 'NVIDIA Corporation', 'NVSMI', 'nvidia-smi.exe')
    if not os.path.isfile(path):
        path = 'nvidia-smi'
    command = [path] + list(args)
    return subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)


def get_gpu_count():
    """GPU数の取得。"""
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpus = os.environ['CUDA_VISIBLE_DEVICES'].strip()
        if gpus == '-1':
            return 0
        return len(np.unique(gpus.split(',')))
    try:
        return len(nvidia_smi('--list-gpus').strip().split('\n'))
    except FileNotFoundError:
        return 0


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
