"""Kerasの損失関数。"""

import numpy as np

from . import backend


def balanced_binary_crossentropy(alpha=0.5):
    """αによるクラス間のバランス補正ありのbinary_crossentropy。

    Focal lossの論文ではα=0.75が良いとされていた。(class 0の重みが0.25)
    """
    def _balanced_binary_crossentropy(y_true, y_pred):
        import keras.backend as K
        a_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        p_t = K.clip(p_t, K.epsilon(), 1 - K.epsilon())
        return -a_t * K.log(p_t)
    return balanced_binary_crossentropy


def binary_focal_loss(alpha=0.25, gamma=2.0):
    """2クラス分類用focal loss (https://arxiv.org/pdf/1708.02002v1.pdf)。"""
    def _binary_focal_loss(y_true, y_pred):
        import keras.backend as K
        a_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        p_t = K.clip(p_t, K.epsilon(), 1 - K.epsilon())
        return -a_t * K.pow(1 - p_t, gamma) * K.log(p_t)
    return _binary_focal_loss


def balanced_categorical_crossentropy(y_true, y_pred, alpha=None):
    """αによるクラス間のバランス補正ありのcategorical_crossentropy。

    Focal lossの論文ではα=0.75が良いとされていた。(class 0の重みが0.25)
    """
    import keras.backend as K
    assert K.image_data_format() == 'channels_last'

    if alpha is None:
        class_weights = -1  # 「-K.sum()」するとpylintが誤検知するのでここに入れ込んじゃう
    else:
        nb_classes = K.int_shape(y_pred)[-1]
        class_weights = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (nb_classes - 1))
        class_weights = np.reshape(class_weights, (1, 1, -1))
        class_weights = -class_weights  # 「-K.sum()」するとpylintが誤検知するのでここに入れ込んじゃう

    y_pred = K.maximum(y_pred, K.epsilon())
    return K.sum(y_true * K.log(y_pred) * class_weights, axis=-1)


def categorical_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """多クラス分類用focal loss (https://arxiv.org/pdf/1708.02002v1.pdf)。"""
    import keras.backend as K

    assert K.image_data_format() == 'channels_last'
    nb_classes = K.int_shape(y_pred)[-1]
    class_weights = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (nb_classes - 1))
    class_weights = np.reshape(class_weights, (1, 1, -1))
    class_weights = -class_weights  # 「-K.sum()」するとpylintが誤検知するのでここに入れ込んじゃう

    y_pred = K.maximum(y_pred, K.epsilon())
    return K.sum(K.pow(1 - y_pred, gamma) * y_true * K.log(y_pred) * class_weights, axis=-1)


def lovasz_hinge(y_true, y_pred):
    """Binary Lovasz hinge loss。"""
    from .lovasz_softmax import lovasz_losses_tf
    logit = backend.logit(y_pred)
    return lovasz_losses_tf.lovasz_hinge(logit, y_true)


def lovasz_hinge_elup1(y_true, y_pred):
    """Binary Lovasz hinge loss。(elu+1)"""
    from .lovasz_softmax import lovasz_losses_tf
    logit = backend.logit(y_pred)
    return lovasz_losses_tf.lovasz_hinge(logit, y_true, hinge_func='elu+1')


def lovasz_hinge_swish(y_true, y_pred):
    """Binary Lovasz hinge loss。(elu+1)"""
    from .lovasz_softmax import lovasz_losses_tf
    logit = backend.logit(y_pred)
    return lovasz_losses_tf.lovasz_hinge(logit, y_true, hinge_func='swish')


def od_bias_initializer(nb_classes, pi=0.01):
    """Object Detectionの最後のクラス分類のbias_initializer。

    nb_classesは背景を含むクラス数。0が背景。
    あるいはnb_classes == 1なら2クラス分類として0が背景、1がオブジェクトとする。
    """
    import keras
    import keras.backend as K

    class FocalLossBiasInitializer(keras.initializers.Initializer):
        """focal loss用の最後のクラス分類のbias_initializer。

        # 引数
        - nb_classes: 背景を含むクラス数。class 0が背景。
        """

        def __init__(self, nb_classes, pi=0.01):
            self.nb_classes = nb_classes
            self.pi = pi

        def __call__(self, shape, dtype=None):
            assert len(shape) == 1
            assert shape[0] % self.nb_classes == 0
            if self.nb_classes == 1:
                bias = -np.log((1 - self.pi) / self.pi)
            else:
                x = np.log(((nb_classes - 1) * (1 - self.pi)) / self.pi)
                bias = [x] + [0] * (nb_classes - 1)  # 背景が0.99%になるような値。21クラス分類なら7.6くらい。(結構大きい…)
                bias = bias * (shape[0] // self.nb_classes)
            return K.constant(bias, shape=shape, dtype=dtype)

        def get_config(self):
            return {'nb_classes': self.nb_classes}

    return FocalLossBiasInitializer(nb_classes, pi)


def l1_smooth_loss(y_true, y_pred):
    """L1-smooth loss。"""
    import keras.backend as K
    import tensorflow as tf
    abs_loss = K.abs(y_true - y_pred)
    sq_loss = 0.5 * K.square(y_true - y_pred)
    l1_loss = tf.where(K.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    l1_loss = K.sum(l1_loss, axis=-1)
    return l1_loss
