"""keras.backendやtensorflowの基礎的な関数など。"""

import tensorflow as tf

from . import K


def clip64(x, epsilon=1e-7):
    """float64にキャストして[epsilon, 1 - epsilon]にclip。"""
    return tf.clip_by_value(K.cast(x, "float64"), epsilon, 1 - epsilon)


def logit(x, epsilon=1e-7):
    """ロジット関数。シグモイド関数の逆関数。

    logit(x) = log(x / (1 - x)) なのだが、
    1 - xのところがx ≒ 1のとき桁落ちするのでfloat64で計算する。

    """
    x = clip64(x, epsilon)
    return K.log(K.cast(x / (1 - x), "float32"))


def binary_crossentropy(y_true, y_pred, from_logits=False, alpha=None):
    """クラス間のバランス補正ありのbinary_crossentropy。"""
    assert alpha is None or 0 <= alpha <= 1

    if not from_logits:
        y_pred = logit(y_pred)

    # 前提知識:
    # -log(sigmoid(x)) = log(1 + exp(-x))
    #                  = -x + log(exp(x) + 1)
    #                  = -x + log1p(exp(x))
    # -log(1 - sigmoid(x)) = log(exp(x) + 1)
    #                      = log1p(exp(x))

    if alpha is None:
        loss = tf.log1p(K.exp(y_pred)) - y_true * y_pred
    else:
        t = 2 * alpha * y_true - alpha - y_true + 1
        loss = 2 * (t * tf.log1p(K.exp(y_pred)) - alpha * y_true * y_pred)

    return loss


def binary_focal_loss(y_true, y_pred, gamma=2.0, from_logits=False, alpha=None):
    """2クラス分類用Focal Loss <https://arxiv.org/abs/1708.02002>。"""
    assert alpha is None or 0 <= alpha <= 1

    if from_logits:
        y_pred, y_logit = K.sigmoid(y_pred), y_pred
    else:
        y_logit = logit(y_pred)

    y_pred_inv = K.cast(1 - clip64(y_pred), "float32")

    # 前提知識:
    # -log(sigmoid(x)) = log(1 + exp(-x))
    #                  = -x + log(exp(x) + 1)
    #                  = -x + log1p(exp(x))
    # -log(1 - sigmoid(x)) = log(exp(x) + 1)
    #                      = log1p(exp(x))

    t = tf.log1p(K.exp(y_logit))
    loss1 = y_true * (y_pred_inv ** gamma) * (-y_logit + t)
    loss2 = (1 - y_true) * (y_pred ** gamma) * t

    if alpha is None:
        loss = loss1 + loss2
    else:
        loss = (2 * alpha) * loss1 + (2 * (1 - alpha)) * loss2

    return loss


def lovasz_weights(y_true, perm, alpha=None):
    """Lovasz hingeなどの損失の重み付け部分。"""
    y_true_sorted = K.gather(y_true, perm)
    y_true_total = K.sum(y_true_sorted)
    inter = y_true_total - tf.cumsum(y_true_sorted)
    union = y_true_total + tf.cumsum(1.0 - y_true_sorted)
    iou = 1.0 - inter / union
    weights = tf.concat((iou[:1], iou[1:] - iou[:-1]), 0)
    if alpha is not None:
        weights *= 2 * (y_true_sorted * alpha + (1 - y_true_sorted) * (1 - alpha))
    return weights
