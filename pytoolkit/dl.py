"""DeepLearning(主にKeras)関連。"""
import functools
import os
import pathlib
import subprocess

import numpy as np
import tensorflow as tf

from . import K, hvd


def wrap_session(config=None, gpu_options=None, use_horovod=False):
    """session()のデコレーター版。"""
    def _decorator(func):
        @functools.wraps(func)
        def _decorated_func(*args, **kwargs):
            with session(config=config, gpu_options=gpu_options, use_horovod=use_horovod):
                return func(*args, **kwargs)
        return _decorated_func
    return _decorator


def session(config=None, gpu_options=None, use_horovod=False):
    """TensorFlowのセッションの初期化・後始末。

    使い方::

        with tk.dl.session():
            # kerasの処理


    Args:
        use_horovod: hvd.init()と、visible_device_listの指定を行う。

    """
    class SessionScope:  # pylint: disable=R0903

        def __init__(self, config=None, gpu_options=None, use_horovod=False):
            self.config = config or {}
            self.gpu_options = gpu_options or {}
            self.use_horovod = use_horovod
            self.session = None

        def __enter__(self):
            if self.use_horovod:
                if hvd.initialized():
                    hvd.barrier()  # 初期化済みなら初期化はしない。念のためタイミングだけ合わせる。
                else:
                    hvd.init()
                if hvd.initialized() and get_gpu_count() > 0:
                    self.gpu_options['visible_device_list'] = str(hvd.get().local_rank())
            if K.backend() == 'tensorflow':
                self.config['allow_soft_placement'] = True
                self.gpu_options['allow_growth'] = True
                if 'OMP_NUM_THREADS' in os.environ and 'intra_op_parallelism_threads' not in self.config:
                    self.config['intra_op_parallelism_threads'] = int(os.environ['OMP_NUM_THREADS'])
                config = tf.ConfigProto(**self.config)
                for k, v in self.gpu_options.items():
                    setattr(config.gpu_options, k, v)
                self.session = tf.Session(config=config)
                K.set_session(self.session)
            return self

        def __exit__(self, *exc_info):
            if K.backend() == 'tensorflow':
                self.session = None
                K.clear_session()

    return SessionScope(config=config, gpu_options=gpu_options, use_horovod=use_horovod)


def get_gpu_count():
    """GPU数の取得。"""
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpus = os.environ['CUDA_VISIBLE_DEVICES'].strip()
        if gpus in ('-1', 'none'):
            return 0
        return len(np.unique(gpus.split(',')))
    try:
        result_text = nvidia_smi('--list-gpus').strip()
        if 'No devices found' in result_text:
            return 0
        return len([l for l in result_text.split('\n') if len(l) > 0])
    except FileNotFoundError:
        return 0


def nvidia_smi(*args):
    """nvidia-smiコマンドを実行する。"""
    path = pathlib.Path(os.environ.get('ProgramFiles', '')) / 'NVIDIA Corporation' / 'NVSMI' / 'nvidia-smi.exe'
    if not path.is_file():
        path = 'nvidia-smi'
    command = [str(path)] + list(args)
    return subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
