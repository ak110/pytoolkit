"""各種ユーティリティ"""
import functools
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np
import sklearn.externals.joblib as joblib

from . import log


def noqa(*args):
    """pylintなどの誤検知対策用の空の関数。"""
    assert args is None or args is not None  # noqa
    assert args is None or args is not None  # noqa


def memorized(cache_path, func):
    """結果をcache_pathにキャッシュする処理

    funcは呼び出すたびに毎回同じ結果が返るものという仮定のもとに単純にキャッシュするだけ。
    """
    if os.path.isfile(cache_path):
        return joblib.load(cache_path)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    result = func()
    joblib.dump(result, cache_path)
    return result


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
    return len(nvidia_smi('--list-gpus').strip().split('\n'))


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
