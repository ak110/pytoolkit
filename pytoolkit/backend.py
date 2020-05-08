"""tf.keras.backendやtensorflowの基礎的な関数など。"""
import typing

import numpy as np
import tensorflow as tf


def clip64(x: tf.Tensor, epsilon=1e-7) -> tf.Tensor:
    """float64にキャストして[epsilon, 1 - epsilon]にclip。"""
    return tf.clip_by_value(tf.cast(x, tf.float64), epsilon, 1 - epsilon)


def logit(x: tf.Tensor, epsilon=1e-7) -> tf.Tensor:
    """ロジット関数。シグモイド関数の逆関数。

    logit(x) = log(x / (1 - x)) なのだが、
    1 - xのところがx ≒ 1のとき桁落ちするのでfloat64で計算する。

    """
    x = clip64(x, epsilon)
    return tf.math.log(tf.cast(x / (1 - x), tf.float32))


def lovasz_weights(y_true, perm, alpha=None):
    """Lovasz hingeなどの損失の重み付け部分。"""
    y_true_sorted = tf.gather(y_true, perm)
    y_true_total = tf.math.reduce_sum(y_true_sorted)
    inter = y_true_total - tf.cumsum(y_true_sorted)
    union = y_true_total + tf.cumsum(1.0 - y_true_sorted)
    iou = 1.0 - inter / union
    weights = tf.concat((iou[:1], iou[1:] - iou[:-1]), 0)
    if alpha is not None:
        weights *= 2 * (y_true_sorted * alpha + (1 - y_true_sorted) * (1 - alpha))
    return weights


def logcosh(x):
    """log(cosh(x))。Smooth L1 lossみたいなもの。"""
    return x + tf.math.softplus(-2.0 * x) - np.log(2.0)


def log_softmax(x, axis=-1):
    """log(softmax(x))"""
    return x - tf.math.reduce_logsumexp(x, axis=axis, keepdims=True)


def reduce_mask(x: tf.Tensor, mask: tf.Tensor, axis: typing.Sequence[int]):
    """加重平均。reduce_sum(x * mask, axis) / reduce_sum(mask, axis)"""
    assert x.shape.rank == mask.shape.rank, f"shape error: x={x} mask={mask}"
    if mask.dtype != x.dtype:
        mask = tf.cast(mask, x.dtype)
    axis = tuple(axis)
    size = tf.math.reduce_sum(mask, axis=axis)
    return tf.math.reduce_sum(x * mask, axis=axis) / tf.math.maximum(size, 1)


def reduce_losses(losses: typing.Sequence[tf.Tensor]) -> tf.Tensor:
    """1次元のtensorを複数受け取り、0次元のtensorを1つ返す。Endpointレイヤー作るとき用。"""
    for x in losses:
        assert x.shape.rank == 1, f"shape error: {x}"
    return tf.math.reduce_mean(tf.stack(losses))
