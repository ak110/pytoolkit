"""DeepLearning(主にKeras)関連。"""
import os

from . import hvd


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    from . import layers, optimizers
    custom_objects = {}
    custom_objects.update(layers.get_custom_objects())
    custom_objects.update(optimizers.get_custom_objects())
    return custom_objects


def session(config=None, gpu_options=None, use_horovod=False):
    """TensorFlowのセッションの初期化・後始末。

    # 使い方

    ```
    with tk.dl.session():

        # kerasの処理

    ```

    # 引数
    - use_horovod: hvd.init()と、visible_device_listの指定を行う。

    """
    import keras.backend as K

    class _Scope(object):  # pylint: disable=R0903

        def __init__(self, config=None, gpu_options=None, use_horovod=False):
            self.config = config or {}
            self.gpu_options = gpu_options or {}
            self.use_horovod = use_horovod

        def __enter__(self):
            if self.use_horovod:
                if hvd.initialized():
                    hvd.barrier()  # 初期化済みなら初期化はしない。念のためタイミングだけ合わせる。
                else:
                    hvd.init()
                if hvd.initialized():
                    self.gpu_options['visible_device_list'] = str(hvd.get().local_rank())
            if K.backend() == 'tensorflow':
                self.config['allow_soft_placement'] = True
                self.gpu_options['allow_growth'] = True
                if 'OMP_NUM_THREADS' in os.environ and 'intra_op_parallelism_threads' not in self.config:
                    self.config['intra_op_parallelism_threads'] = int(os.environ['OMP_NUM_THREADS'])
                import tensorflow as tf
                config = tf.ConfigProto(**self.config)
                for k, v in self.gpu_options.items():
                    setattr(config.gpu_options, k, v)
                K.set_session(tf.Session(config=config))
            return self

        def __exit__(self, *exc_info):
            if self.use_horovod and hvd.initialized():
                hvd.barrier()  # 念のためタイミングを合わせる。
            if K.backend() == 'tensorflow':
                K.clear_session()

    return _Scope(config=config, gpu_options=gpu_options, use_horovod=use_horovod)


def device(cpu=False, gpu=False):
    """TensorFlowのデバイス指定の簡単なラッパー。"""
    assert cpu != gpu
    import tensorflow as tf
    if cpu:
        return tf.device('/cpu:0')
    else:
        return tf.device('/gpu:0')
