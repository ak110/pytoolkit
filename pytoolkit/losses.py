"""Kerasの損失関数。"""

import numpy as np
import tensorflow as tf

from .. import pytoolkit as tk
from . import K


def binary_crossentropy(y_true, y_pred, from_logits=False, alpha=None):
    """クラス間のバランス補正ありのbinary_crossentropy。

    Args:
        alpha (float or None): class 1の重み。

    """
    loss = tk.backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits, alpha=alpha)
    return K.sum(loss, axis=list(range(1, K.ndim(y_true))))


def binary_focal_loss(y_true, y_pred, gamma=2.0, from_logits=False, alpha=None):
    """2クラス分類用Focal Loss <https://arxiv.org/abs/1708.02002>。

    Args:
        alpha (float or None): class 1の重み。論文では0.25。

    """
    loss = tk.backend.binary_focal_loss(y_true, y_pred, gamma=gamma, from_logits=from_logits, alpha=alpha)
    return K.sum(loss, axis=list(range(1, K.ndim(y_true))))


def categorical_crossentropy(y_true, y_pred, alpha=None, class_weights=None):
    """クラス間のバランス補正ありのcategorical_crossentropy。

    Focal lossの論文ではα=0.75が良いとされていた。(class 0の重みが0.25)
    """
    assert alpha is None or class_weights is None  # 両方同時の指定はNG
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
    return -K.sum(y_true * K.log(y_pred) * cw, axis=list(range(1, K.ndim(y_true))))


def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
    """多クラス分類用Focal Loss <https://arxiv.org/abs/1708.02002>。

    Args:
        alpha (float or None): class 0以外の重み。論文では0.25。

    """
    assert K.image_data_format() == 'channels_last'
    if alpha is None:
        class_weights = 1
    else:
        nb_classes = K.int_shape(y_pred)[-1]
        class_weights = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (nb_classes - 1))
        class_weights = np.reshape(class_weights, (1, 1, -1))

    y_pred = K.maximum(y_pred, K.epsilon())
    return -K.sum(K.pow(1 - y_pred, gamma) * y_true * K.log(y_pred) * class_weights, axis=list(range(1, K.ndim(y_true))))  # pylint: disable=invalid-unary-operand-type


def lovasz_hinge(y_true, y_pred, from_logits=False, per_sample=True, activation='elu+1'):
    """Lovasz hinge loss。<https://arxiv.org/abs/1512.07797>"""
    if not from_logits:
        y_pred = tk.backend.logit(y_pred)
    if per_sample:
        def loss_per_sample(elems):
            yt, yp = elems
            return lovasz_hinge(yt, yp, from_logits=True, per_sample=False, activation=activation)
        return tf.map_fn(loss_per_sample, (y_true, y_pred), dtype=tf.float32)

    y_true = K.reshape(y_true, (-1,))
    y_pred = K.reshape(y_pred, (-1,))
    signs = y_true * 2.0 - 1.0  # -1 ～ +1
    errors = 1.0 - y_pred * signs
    errors_sorted, perm = tf.nn.top_k(errors, k=K.shape(errors)[0])
    weights = tk.backend.lovasz_weights(y_true, perm)
    if activation == 'relu':
        errors_sorted = tf.nn.relu(errors_sorted)
    elif activation == 'elu+1':
        errors_sorted = tf.nn.elu(errors_sorted) + 1
    else:
        raise ValueError(f'Invalid activation: {activation}')
    loss = tf.tensordot(errors_sorted, tf.stop_gradient(weights), 1)
    assert K.ndim(loss) == 0
    return loss


def lovasz_binary_crossentropy(y_true, y_pred, from_logits=False, per_sample=True, epsilon=0.01, alpha=None):
    """Lovasz hinge lossのhingeじゃない版。

    Args:
        epsilon (float): sigmoidの値をclipする値。 sigmoid=0.01のときlogit=-4.6くらい。

    """
    if per_sample:
        def loss_per_sample(elems):
            yt, yp = elems
            return lovasz_binary_crossentropy(yt, yp, from_logits=from_logits, per_sample=False, epsilon=epsilon, alpha=alpha)
        return tf.map_fn(loss_per_sample, (y_true, y_pred), dtype=tf.float32)

    y_true = K.reshape(y_true, (-1,))
    y_pred = K.reshape(y_pred, (-1,))
    if from_logits:
        lpsilon = tk.math.logit(epsilon)
        y_pred = K.clip(y_pred, lpsilon, -lpsilon)
    else:
        y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
    errors = tk.backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    errors_sorted, perm = tf.nn.top_k(errors, k=K.shape(errors)[0])
    weights = tk.backend.lovasz_weights(y_true, perm, alpha=alpha)
    loss = tf.tensordot(errors_sorted, tf.stop_gradient(weights), 1)
    assert K.ndim(loss) == 0
    return loss


def lovasz_softmax(y_true, y_pred, per_sample=True):
    """Lovasz softmax loss。<https://arxiv.org/abs/1705.08790>"""
    if per_sample:
        def loss_per_sample(elems):
            yt, yp = elems
            return lovasz_softmax(yt, yp, per_sample=False)
        return tf.map_fn(loss_per_sample, (y_true, y_pred), dtype=tf.float32)

    num_classes = K.int_shape(y_true)[-1]
    y_pred = K.reshape(y_pred, (-1, num_classes))
    y_true = K.reshape(y_true, (-1, num_classes))
    losses = []
    for c in range(num_classes):
        errors = K.abs(y_true[:, c] - y_pred[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=K.shape(errors)[0])
        weights = tk.backend.lovasz_weights(y_true[:, c], perm)
        loss = tf.tensordot(errors_sorted, tf.stop_gradient(weights), 1)
        losses.append(loss)
    return tf.reduce_mean(losses)


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
