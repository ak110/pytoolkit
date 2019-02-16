"""Kerasの損失関数。"""

import numpy as np
import tensorflow as tf

from . import K, backend
from .lovasz_softmax import lovasz_losses_tf


def make_binary_crossentropy(alpha=0.5):
    """クラス間のバランス補正ありのbinary_crossentropy。

    Focal lossの論文ではα=0.75が良いとされていた。(class 0の重みが0.25)
    """
    def binary_crossentropy(y_true, y_pred):
        a_t = y_true * alpha + (1 - y_true) * (1 - alpha) if alpha is not None else 1
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        p_t = K.clip(p_t, K.epsilon(), 1 - K.epsilon())
        return -a_t * K.log(p_t)
    return binary_crossentropy


def make_binary_focal_loss(alpha=0.25, gamma=2.0):
    """2クラス分類用focal loss (https://arxiv.org/pdf/1708.02002v1.pdf)。"""
    def binary_focal_loss(y_true, y_pred):
        a_t = y_true * alpha + (1 - y_true) * (1 - alpha) if alpha is not None else 1
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        p_t = K.clip(p_t, K.epsilon(), 1 - K.epsilon())
        return -a_t * K.pow(1 - p_t, gamma) * K.log(p_t)
    return binary_focal_loss


def make_categorical_crossentropy(alpha=None, class_weights=None):
    """クラス間のバランス補正ありのcategorical_crossentropy。

    Focal lossの論文ではα=0.75が良いとされていた。(class 0の重みが0.25)
    """
    assert alpha is None or class_weights is None  # 両方同時の指定はNG

    def categorical_crossentropy(y_true, y_pred):
        assert K.image_data_format() == 'channels_last'

        if alpha is None:
            if class_weights is None:
                cw = 1
            else:
                cw = np.reshape(class_weights, (1, 1, -1))
        else:
            num_classes = K.int_shape(y_pred)[-1]
            cw = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (num_classes - 1))
            cw = np.reshape(cw, (1, 1, -1))

        y_pred = K.maximum(y_pred, K.epsilon())
        return -K.sum(y_true * K.log(y_pred) * cw, axis=-1)
    return categorical_crossentropy


def make_categorical_focal_loss(alpha=0.25, gamma=2.0):
    """多クラス分類用focal loss (https://arxiv.org/pdf/1708.02002v1.pdf)。"""
    def categorical_focal_loss(y_true, y_pred):
        assert K.image_data_format() == 'channels_last'
        if alpha is None:
            class_weights = 1
        else:
            nb_classes = K.int_shape(y_pred)[-1]
            class_weights = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (nb_classes - 1))
            class_weights = np.reshape(class_weights, (1, 1, -1))

        y_pred = K.maximum(y_pred, K.epsilon())
        return -K.sum(K.pow(1 - y_pred, gamma) * y_true * K.log(y_pred) * class_weights, axis=-1)  # pylint: disable=invalid-unary-operand-type
    return categorical_focal_loss


def lovasz_hinge(y_true, y_pred):
    """Binary Lovasz hinge loss。"""
    logit = backend.logit(y_pred)
    return lovasz_losses_tf.lovasz_hinge(logit, y_true)


def lovasz_hinge_elup1(y_true, y_pred):
    """Binary Lovasz hinge loss。(elu+1)"""
    logit = backend.logit(y_pred)
    return lovasz_losses_tf.lovasz_hinge(logit, y_true, hinge_func='elu+1')


def symmetric_lovasz_hinge_elup1(y_true, y_pred):
    """Binary Lovasz hinge lossの0, 1対称版。(elu+1)"""
    logit = backend.logit(y_pred)
    loss1 = lovasz_losses_tf.lovasz_hinge(logit, y_true, hinge_func='elu+1')
    loss2 = lovasz_losses_tf.lovasz_hinge(-logit, 1 - y_true, hinge_func='elu+1')
    return (loss1 + loss2) / 2


def make_lovasz_softmax(ignore=None):
    """Lovasz softmax loss。"""
    def _lovasz_softmax(y_true, y_pred):
        return lovasz_losses_tf.lovasz_softmax(y_pred, K.argmax(y_true, axis=-1), ignore=ignore)
    return _lovasz_softmax


def make_mixed_lovasz_softmax(ignore=None):
    """Lovasz softmax loss + CE。"""
    def _lovasz_softmax(y_true, y_pred):
        loss1 = lovasz_losses_tf.lovasz_softmax(y_pred, K.argmax(y_true, axis=-1), ignore=ignore)
        loss2 = K.categorical_crossentropy(y_true, y_pred)
        return loss1 * 0.9 + loss2 * 0.1
    return _lovasz_softmax


def l1_smooth_loss(y_true, y_pred):
    """L1-smooth loss。"""
    abs_loss = K.abs(y_true - y_pred)
    sq_loss = 0.5 * K.square(y_true - y_pred)
    l1_loss = tf.where(K.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    l1_loss = K.sum(l1_loss, axis=-1)
    return l1_loss


def mse(y_true, y_pred):
    """AutoEncoderとか用mean squared error"""
    return K.mean(K.square(y_pred - y_true), axis=list(range(1, K.ndim(y_true))))


def mae(y_true, y_pred):
    """AutoEncoderとか用mean absolute error"""
    return K.mean(K.abs(y_pred - y_true), axis=list(range(1, K.ndim(y_true))))


def rmse(y_true, y_pred):
    """AutoEncoderとか用root mean squared error"""
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=list(range(1, K.ndim(y_true)))))
