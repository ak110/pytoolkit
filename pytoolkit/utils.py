"""各種ユーティリティ"""
import multiprocessing as mp
import os
import subprocess

import numpy as np
import sklearn.externals.joblib


def create_tee_logger(output_path, name=None, append=False,
                      rotate=False, max_bytes=1048576, backup_count=10,
                      fmt='%(asctime)s [%(levelname)-5s] %(message)s'):
    """標準出力とファイルに内容を出力するloggerを作成して返す。"""
    from . import log
    logger = log.get(name=name)
    logger.addHandler(log.stream_handler(fmt=fmt))
    if output_path is not None:
        logger.addHandler(log.file_handler(output_path, append, rotate, max_bytes, backup_count, fmt=fmt))
    return logger


def memorized(cache_path, func):
    """結果をcache_pathにキャッシュする処理

    funcは呼び出すたびに毎回同じ結果が返るものという仮定のもとに単純にキャッシュするだけ。
    """
    if os.path.isfile(cache_path):
        return sklearn.externals.joblib.load(cache_path)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    result = func()
    sklearn.externals.joblib.dump(result, cache_path)
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
