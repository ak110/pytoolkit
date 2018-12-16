"""Kerasの損失関数。"""

import numpy as np

from . import backend


def balanced_binary_crossentropy(alpha=0.5):
    """αによるクラス間のバランス補正ありのbinary_crossentropy。

    Focal lossの論文ではα=0.75が良いとされていた。(class 0の重みが0.25)
    """
    def _balanced_binary_crossentropy(y_true, y_pred):
        import tensorflow as tf
        a_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        p_t = tf.keras.backend.clip(p_t, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return -a_t * tf.keras.backend.log(p_t)
    return balanced_binary_crossentropy


def binary_focal_loss(alpha=0.25, gamma=2.0):
    """2クラス分類用focal loss (https://arxiv.org/pdf/1708.02002v1.pdf)。"""
    def _binary_focal_loss(y_true, y_pred):
        import tensorflow as tf
        a_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        p_t = tf.keras.backend.clip(p_t, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return -a_t * tf.keras.backend.pow(1 - p_t, gamma) * tf.keras.backend.log(p_t)
    return _binary_focal_loss


def balanced_categorical_crossentropy(y_true, y_pred, alpha=None):
    """αによるクラス間のバランス補正ありのcategorical_crossentropy。

    Focal lossの論文ではα=0.75が良いとされていた。(class 0の重みが0.25)
    """
    import tensorflow as tf
    assert tf.keras.backend.image_data_format() == 'channels_last'

    if alpha is None:
        class_weights = -1  # 「-tf.keras.backend.sum()」するとpylintが誤検知するのでここに入れ込んじゃう
    else:
        nb_classes = tf.keras.backend.int_shape(y_pred)[-1]
        class_weights = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (nb_classes - 1))
        class_weights = np.reshape(class_weights, (1, 1, -1))
        class_weights = -class_weights  # 「-tf.keras.backend.sum()」するとpylintが誤検知するのでここに入れ込んじゃう

    y_pred = tf.keras.backend.maximum(y_pred, tf.keras.backend.epsilon())
    return tf.keras.backend.sum(y_true * tf.keras.backend.log(y_pred) * class_weights, axis=-1)


def categorical_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """多クラス分類用focal loss (https://arxiv.org/pdf/1708.02002v1.pdf)。"""
    import tensorflow as tf
    assert tf.keras.backend.image_data_format() == 'channels_last'

    nb_classes = tf.keras.backend.int_shape(y_pred)[-1]
    class_weights = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (nb_classes - 1))
    class_weights = np.reshape(class_weights, (1, 1, -1))
    class_weights = -class_weights  # 「-tf.keras.backend.sum()」するとpylintが誤検知するのでここに入れ込んじゃう

    y_pred = tf.keras.backend.maximum(y_pred, tf.keras.backend.epsilon())
    return tf.keras.backend.sum(tf.keras.backend.pow(1 - y_pred, gamma) * y_true * tf.keras.backend.log(y_pred) * class_weights, axis=-1)


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


def symmetric_lovasz_hinge_elup1(y_true, y_pred):
    """Binary Lovasz hinge lossの0, 1対称版。(elu+1)"""
    from .lovasz_softmax import lovasz_losses_tf
    logit = backend.logit(y_pred)
    loss1 = lovasz_losses_tf.lovasz_hinge(logit, y_true, hinge_func='elu+1')
    loss2 = lovasz_losses_tf.lovasz_hinge(-logit, 1 - y_true, hinge_func='elu+1')
    return (loss1 + loss2) / 2


def make_lovasz_softmax(ignore=None):
    """Lovasz softmax loss。"""
    def _lovasz_softmax(y_true, y_pred):
        import tensorflow as tf
        from .lovasz_softmax import lovasz_losses_tf
        return lovasz_losses_tf.lovasz_softmax(y_pred, tf.keras.backend.argmax(y_true, axis=-1), ignore=ignore)
    return _lovasz_softmax


def make_mixed_lovasz_softmax(ignore=None):
    """Lovasz softmax loss + CE。"""
    def _lovasz_softmax(y_true, y_pred):
        import tensorflow as tf
        from .lovasz_softmax import lovasz_losses_tf
        loss1 = lovasz_losses_tf.lovasz_softmax(y_pred, tf.keras.backend.argmax(y_true, axis=-1), ignore=ignore)
        loss2 = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
        return loss1 * 0.9 + loss2 * 0.1
    return _lovasz_softmax


def l1_smooth_loss(y_true, y_pred):
    """L1-smooth loss。"""
    import tensorflow as tf
    abs_loss = tf.keras.backend.abs(y_true - y_pred)
    sq_loss = 0.5 * tf.keras.backend.square(y_true - y_pred)
    l1_loss = tf.where(tf.keras.backend.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    l1_loss = tf.keras.backend.sum(l1_loss, axis=-1)
    return l1_loss
