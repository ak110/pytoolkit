"""Kerasの損失関数。"""

import numpy as np


def get_custom_objects():
    """独自オブジェクトのdictを返す。"""
    classes = [
        focal_loss_bias_initializer(),
    ]
    return {c.__name__: c for c in classes}


def focal_loss_bias_initializer():
    """focal loss用の最後のクラス分類のbias_initializer。

    # 引数
    - nb_classes: 背景を含むクラス数。class 0が背景。
    - pi: π。前景の重み。
    """
    import tensorflow as tf

    class FocalLossBiasInitializer(tf.keras.initializers.Initializer):
        """focal loss用の最後のクラス分類のbias_initializer。

        # 引数
        - nb_classes: 背景を含むクラス数。class 0が背景。
        - pi: π。前景の重み。
        """

        def __init__(self, nb_classes, pi=0.01):
            self.nb_classes = nb_classes
            self.pi = pi

        def __call__(self, shape, dtype=None, partition_info=None):
            assert len(shape) == 1
            assert shape[0] % self.nb_classes == 0
            if self.nb_classes == 1:
                bias = -np.log((1 - self.pi) / self.pi)
            else:
                x = np.log(((self.nb_classes - 1) * (1 - self.pi)) / self.pi)
                bias = [x] + [0] * (self.nb_classes - 1)  # 背景が0.99%になるような値。21クラス分類なら7.6くらい。(結構大きい…)
                bias = bias * (shape[0] // self.nb_classes)
            return tf.keras.backend.constant(bias, shape=shape, dtype=dtype)

        def get_config(self):
            return {
                'nb_classes': self.nb_classes,
                'pi': self.pi,
            }

    return FocalLossBiasInitializer
