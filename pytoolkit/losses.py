"""Kerasの損失関数を実装するための関数など。"""

import numpy as np
import tensorflow as tf

import pytoolkit as tk

K = tf.keras.backend


def reduce(x, reduce_mode):
    """バッチ次元だけ残して次元を削減する。"""
    if reduce_mode is None:
        return x
    axes = list(range(1, K.ndim(x)))
    if len(axes) <= 0:
        return x
    return {"sum": K.sum, "mean": K.mean}[reduce_mode](x, axis=axes)


def binary_crossentropy(
    y_true, y_pred, from_logits=False, alpha=None, reduce_mode="sum"
):
    """クラス間のバランス補正ありのbinary_crossentropy。

    Args:
        alpha (float or None): class 1の重み。

    """
    assert alpha is None or 0 <= alpha <= 1

    if not from_logits:
        y_pred = tk.backend.logit(y_pred)

    # 前提知識:
    # -log(sigmoid(x)) = log(1 + exp(-x))
    #                  = -x + log(exp(x) + 1)
    #                  = -x + log1p(exp(x))
    # -log(1 - sigmoid(x)) = log(exp(x) + 1)
    #                      = log1p(exp(x))

    if alpha is None:
        loss = tf.math.log1p(K.exp(y_pred)) - y_true * y_pred
    else:
        t = 2 * alpha * y_true - alpha - y_true + 1
        loss = 2 * (t * tf.math.log1p(K.exp(y_pred)) - alpha * y_true * y_pred)

    return reduce(loss, reduce_mode)


def binary_focal_loss(
    y_true, y_pred, gamma=2.0, from_logits=False, alpha=None, reduce_mode="sum"
):
    """2クラス分類用Focal Loss <https://arxiv.org/abs/1708.02002>。

    Args:
        alpha (float or None): class 1の重み。論文では0.25。

    """
    assert alpha is None or 0 <= alpha <= 1

    if from_logits:
        y_pred, y_logit = K.sigmoid(y_pred), y_pred
    else:
        y_logit = tk.backend.logit(y_pred)

    y_pred_inv = K.cast(1 - tk.backend.clip64(y_pred), "float32")

    # 前提知識:
    # -log(sigmoid(x)) = log(1 + exp(-x))
    #                  = -x + log(exp(x) + 1)
    #                  = -x + log1p(exp(x))
    # -log(1 - sigmoid(x)) = log(exp(x) + 1)
    #                      = log1p(exp(x))

    t = tf.math.log1p(K.exp(y_logit))
    loss1 = y_true * (y_pred_inv ** gamma) * (-y_logit + t)
    loss2 = (1 - y_true) * (y_pred ** gamma) * t

    if alpha is None:
        loss = loss1 + loss2
    else:
        loss = (2 * alpha) * loss1 + (2 * (1 - alpha)) * loss2

    return reduce(loss, reduce_mode)


def categorical_crossentropy(
    y_true,
    y_pred,
    from_logits=False,
    alpha=None,
    class_weights=None,
    label_smoothing=None,
    reduce_mode="sum",
):
    """クラス間のバランス補正ありのcategorical_crossentropy。

    Focal lossの論文ではα=0.75が良いとされていた。(class 0の重みが0.25)

    label_smoothing を使う場合は0.2とかを指定。

    References:
        - label smoothing <https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/>

    """
    assert alpha is None or class_weights is None  # 両方同時の指定はNG
    assert K.image_data_format() == "channels_last"

    if alpha is None:
        if class_weights is None:
            cw = 1
        else:
            cw = np.reshape(class_weights, (1, 1, -1))
    else:
        num_classes = y_pred.shape[-1]
        cw = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (num_classes - 1))
        cw = np.reshape(cw, (1, 1, -1))

    if from_logits:
        log_p = tk.backend.log_softmax(y_pred)
    else:
        y_pred = K.maximum(y_pred, K.epsilon())
        log_p = K.log(y_pred)

    loss = -K.sum(y_true * log_p * cw, axis=-1)

    if label_smoothing is not None:
        kl = -K.mean(log_p, axis=-1)
        loss = (1 - label_smoothing) * loss + label_smoothing * kl

    return reduce(loss, reduce_mode)


def categorical_focal_loss(
    y_true, y_pred, from_logits=False, gamma=2.0, alpha=None, reduce_mode="sum"
):
    """多クラス分類用Focal Loss <https://arxiv.org/abs/1708.02002>。

    Args:
        alpha (float or None): class 0以外の重み。論文では0.25。

    """
    assert K.image_data_format() == "channels_last"
    if alpha is None:
        cw = 1
    else:
        num_classes = y_pred.shape[-1]
        cw = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (num_classes - 1))
        cw = np.reshape(cw, (1, 1, -1))

    if from_logits:
        y_pred = tf.nn.softmax(y_pred)

    y_pred = K.maximum(y_pred, K.epsilon())
    loss = -cw * K.pow(1 - y_pred, gamma) * y_true * K.log(y_pred)
    return reduce(loss, reduce_mode)


def symmetric_lovasz_hinge(
    y_true, y_pred, from_logits=False, per_sample=True, activation="elu+1"
):
    """lovasz_hingeのsymmetricバージョン"""
    if not from_logits:
        y_pred = tk.backend.logit(y_pred)
        from_logits = True
    loss1 = lovasz_hinge(y_true, y_pred, from_logits, per_sample, activation)
    loss2 = lovasz_hinge(1 - y_true, -y_pred, from_logits, per_sample, activation)
    return (loss1 + loss2) / 2


def lovasz_hinge(
    y_true, y_pred, from_logits=False, per_sample=True, activation="elu+1"
):
    """Lovasz hinge loss。<https://arxiv.org/abs/1512.07797>"""
    if not from_logits:
        y_pred = tk.backend.logit(y_pred)
        from_logits = True
    if per_sample:

        def loss_per_sample(elems):
            yt, yp = elems
            return lovasz_hinge(
                yt, yp, from_logits=from_logits, per_sample=False, activation=activation
            )

        return tf.map_fn(loss_per_sample, (y_true, y_pred), dtype=tf.float32)

    y_true = K.reshape(y_true, (-1,))
    y_pred = K.reshape(y_pred, (-1,))
    signs = y_true * 2.0 - 1.0  # -1 ～ +1
    errors = 1.0 - y_pred * signs
    errors_sorted, perm = tf.nn.top_k(errors, k=K.shape(errors)[0])
    weights = tk.backend.lovasz_weights(y_true, perm)
    if activation == "relu":
        errors_sorted = tf.nn.relu(errors_sorted)
    elif activation == "elu+1":
        errors_sorted = tf.nn.elu(errors_sorted) + 1
    else:
        raise ValueError(f"Invalid activation: {activation}")
    loss = tf.tensordot(errors_sorted, weights, 1)
    assert K.ndim(loss) == 0
    return loss


def lovasz_binary_crossentropy(
    y_true, y_pred, from_logits=False, per_sample=True, epsilon=0.01, alpha=None
):
    """Lovasz hinge lossのhingeじゃない版。

    Args:
        epsilon: sigmoidの値をclipする値。 sigmoid=0.01のときlogit=-4.6くらい。

    """
    if per_sample:

        def loss_per_sample(elems):
            yt, yp = elems
            return lovasz_binary_crossentropy(
                yt,
                yp,
                from_logits=from_logits,
                per_sample=False,
                epsilon=epsilon,
                alpha=alpha,
            )

        return tf.map_fn(loss_per_sample, (y_true, y_pred), dtype=tf.float32)

    y_true = K.reshape(y_true, (-1,))
    y_pred = K.reshape(y_pred, (-1,))
    if from_logits:
        lpsilon = tk.math.logit(epsilon)
        logits = K.clip(y_pred, lpsilon, -lpsilon)
        y_pred = tf.math.sigmoid(y_pred)
    else:
        y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
        logits = tk.backend.logit(y_pred)
    base_errors = tf.math.abs(y_true - y_pred)
    errors = tf.math.log1p(K.exp(logits)) - y_true * logits  # bce
    _, perm = tf.nn.top_k(errors, k=K.shape(errors)[0])
    errors_sorted = tf.gather(base_errors, perm)
    weights = tk.backend.lovasz_weights(y_true, perm, alpha=alpha)
    loss = tf.tensordot(errors_sorted, weights, 1)
    assert K.ndim(loss) == 0
    return loss


def lovasz_categorical_crossentropy(
    y_true, y_pred, from_logits=False, per_sample=True, epsilon=0.01, alpha=None
):
    """Lovasz Softmaxのsoftmaxじゃない版。"""
    if per_sample:

        def loss_per_sample(elems):
            yt, yp = elems
            return lovasz_categorical_crossentropy(
                yt,
                yp,
                from_logits=from_logits,
                per_sample=False,
                epsilon=epsilon,
                alpha=alpha,
            )

        return tf.map_fn(loss_per_sample, (y_true, y_pred), dtype=tf.float32)

    num_classes = y_pred.shape[-1]
    y_pred = K.reshape(y_pred, (-1, num_classes))
    y_true = K.reshape(y_true, (-1, num_classes))
    if from_logits:
        log_p = tk.backend.log_softmax(y_pred)
    else:
        y_pred = K.maximum(y_pred, K.epsilon())
        log_p = K.log(y_pred)
    errors = -y_true * log_p  # cce
    losses = []
    for c in range(num_classes):
        errors_sorted, perm = tf.nn.top_k(errors[:, c], k=K.shape(errors)[0])
        weights = tk.backend.lovasz_weights(y_true[:, c], perm, alpha=alpha)
        loss = tf.tensordot(errors_sorted, weights, 1)
        assert K.ndim(loss) == 0
        losses.append(loss)
    return loss


def l1_smooth_loss(y_true, y_pred):
    """L1-smooth loss。"""
    abs_loss = K.abs(y_true - y_pred)
    sq_loss = 0.5 * K.square(y_true - y_pred)
    l1_loss = tf.where(K.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    l1_loss = K.sum(l1_loss, axis=-1)
    return l1_loss


def mse(y_true, y_pred, reduce_mode="mean"):
    """mean squared error。"""
    return reduce(K.square(y_pred - y_true), reduce_mode)


def mae(y_true, y_pred, reduce_mode="mean"):
    """mean absolute error。"""
    return reduce(K.abs(y_pred - y_true), reduce_mode)


def rmse(y_true, y_pred, reduce_mode="mean"):
    """root mean squared error。"""
    return K.sqrt(reduce(K.square(y_pred - y_true), reduce_mode))


def mape(y_true, y_pred, reduce_mode="mean"):
    """mean absolute percentage error。"""
    return reduce(K.abs((y_true - y_pred) / y_true), reduce_mode)
