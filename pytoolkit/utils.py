"""各種ユーティリティ"""
import distutils.version
import functools
import logging
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np


def noqa(*args):
    """pylintなどの誤検知対策用の空の関数。"""
    assert args is None or args is not None  # noqa


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


def moving_average(arr, window_size):
    """移動平均"""
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='valid')


def nvidia_smi(*args):
    """nvidia-smiコマンドを実行する。"""
    path = os.path.join(
        os.environ.get('ProgramFiles', ''), 'NVIDIA Corporation', 'NVSMI', 'nvidia-smi.exe')
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


def create_gpu_pool(n_gpus=None, processes_per_gpu=1, initializer=None, initargs=None):
    """CUDA_VISIBLE_DEVICESを指定したGPU数×processes_per_gpu個のプロセスが動くmp.Poolを作って返す。

    ついでにGPU_POOL_PIDという環境変数にプロセスの一意な番号も入れておく。
    (主に processes_per_gpu > 1 のとき用)
    """
    if n_gpus is None:
        n_gpus = get_gpu_count()
    mpq = mp.SimpleQueue()
    for gpu_id in range(n_gpus):
        for process_id in range(processes_per_gpu):
            mpq.put((gpu_id, gpu_id * processes_per_gpu + process_id))
    return mp.Pool(n_gpus * processes_per_gpu, _init_gpu_pool, [mpq, initializer, initargs])


def _init_gpu_pool(mpq, initializer, initargs):
    """初期化"""
    gpu_id, process_id = mpq.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['GPU_POOL_PID'] = str(process_id)  # 一意な番号。たぶんあった方がいいと思うのでついでに。
    if initializer:
        if initargs is None:
            initargs = []
        initializer(*initargs)


def capture_output():
    """stdoutとstderrをキャプチャして戻り値として返すデコレーター"""
    import io

    def _decorator(func):
        @functools.wraps(func)
        def _decorated_func(*args, **kwargs):
            stdout = sys.stdout
            stderr = sys.stderr
            buf = io.StringIO()  # kerasも黙らせる。。
            sys.stdout = buf
            sys.stderr = buf
            try:
                result = func(*args, **kwargs)
                assert result is None
                return buf.getvalue()
            finally:
                sys.stderr = stderr
                sys.stdout = stdout
        return _decorated_func
    return _decorator


def tqdm(iterable=None, desc=None, total=None, leave=True, **kwargs):
    """`tqdm`の簡単なラッパー。"""
    from tqdm import tqdm as t
    return t(iterable, desc, total, leave, ascii=True, ncols=100, **kwargs)


def delayed(fn):
    """joblib.delayedのDeprecationWarning対策。"""
    import sklearn.externals.joblib as joblib
    if distutils.version.LooseVersion(joblib.__version__) >= distutils.version.LooseVersion('0.12'):
        return joblib.delayed(fn)
    else:
        return joblib.delayed(fn, check_pickle=False)


def better_exceptions():
    """`better_exceptions`を有効にする。"""
    try:
        # サブプロセスとか用に環境変数を設定
        os.environ['BETTER_EXCEPTIONS'] = '1'
        # 今のスレッドでアクティブにする
        import better_exceptions as be  # pip install better_exceptions
        be.hook()
    except BaseException:
        logger = logging.getLogger(__name__)
        logger.warning('better_exceptions error', exc_info=True)
