"""DeepLearning(主にKeras)関連。"""
import os


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    from . import layers, optimizers
    return {
        'GroupNormalization': layers.group_normalization(),
        'Destandarization': layers.destandarization(),
        'StocasticAdd': layers.stocastic_add(),
        'NormalNoise': layers.normal_noise(),
        'L2Normalization': layers.l2normalization(),
        'WeightedMean': layers.weighted_mean(),
        'NSGD': optimizers.nsgd(),
    }


def session(config=None, gpu_options=None):
    """TensorFlowのセッションの初期化・後始末。

    # 使い方

    ```
    with tk.dl.session():

        # kerasの処理

    ```

    """
    import keras.backend as K

    class _Scope(object):  # pylint: disable=R0903

        def __init__(self, config=None, gpu_options=None):
            self.config = config or {}
            self.gpu_options = gpu_options or {}

        def __enter__(self):
            if K.backend() == 'tensorflow':
                import tensorflow as tf
                self.config.update({'allow_soft_placement': True})
                self.gpu_options.update({'allow_growth': True})
                if 'OMP_NUM_THREADS' in os.environ and 'intra_op_parallelism_threads' not in self.config:
                    self.config['intra_op_parallelism_threads'] = int(os.environ['OMP_NUM_THREADS'])
                config = tf.ConfigProto(**self.config)
                for k, v in self.gpu_options.items():
                    setattr(config.gpu_options, k, v)
                K.set_session(tf.Session(config=config))

        def __exit__(self, *exc_info):
            if K.backend() == 'tensorflow':
                K.clear_session()

    return _Scope(config=config, gpu_options=gpu_options)


def device(cpu=False, gpu=False):
    """TensorFlowのデバイス指定の簡単なラッパー。"""
    assert cpu != gpu
    import tensorflow as tf
    if cpu:
        return tf.device('/cpu:0')
    else:
        return tf.device('/gpu:0')
