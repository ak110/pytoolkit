"""Kerasのmetrics関連。"""

import tensorflow as tf

from . import K


def binary_accuracy(y_true, y_pred):
    """Soft-targetとかでも一応それっぽい値を返すbinary accuracy。"""
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1)


def binary_iou(y_true, y_pred, target_classes=None, threshold=0.5):
    """画像ごとクラスごとのIoUを算出して平均するmetric。

    Args:
        target_classes: 対象のクラスindexの配列。Noneなら全クラス。
        threshold: 予測の閾値

    """
    if target_classes is not None:
        y_true = y_true[..., target_classes]
        y_pred = y_pred[..., target_classes]
    axes = list(range(1, K.ndim(y_true) - 1))
    t = K.greater_equal(y_true, 0.5)
    p = K.greater_equal(y_pred, threshold)
    inter = K.sum(K.cast(tf.math.logical_and(t, p, name='inter'), 'float32'), axis=axes)
    union = K.sum(K.cast(tf.math.logical_or(t, p, name='union'), 'float32'), axis=axes)
    return inter / K.maximum(union, 1)


def categorical_iou(y_true, y_pred, target_classes=None, strict=True):
    """画像ごとクラスごとのIoUを算出して平均するmetric。

    Args:
        target_classes: 対象のクラスindexの配列。Noneなら全クラス。
        strict: ラベルに無いクラスを予測してしまった場合に減点されるようにするならTrue、ラベルにあるクラスのみ対象にするならFalse。

    """
    axes = list(range(1, K.ndim(y_true) - 1))
    y_classes = K.argmax(y_true, axis=-1)
    p_classes = K.argmax(y_pred, axis=-1)
    active_list = []
    iou_list = []
    for c in target_classes or range(K.int_shape(y_true)[-1]):
        with tf.name_scope(f'class_{c}'):
            y_c = K.equal(y_classes, c)
            p_c = K.equal(p_classes, c)
            inter = K.sum(K.cast(tf.math.logical_and(y_c, p_c, name='inter'), 'float32'), axis=axes)
            union = K.sum(K.cast(tf.math.logical_or(y_c, p_c, name='union'), 'float32'), axis=axes)
            active = union > 0 if strict else K.any(y_c, axis=axes)
            iou = inter / (union + K.epsilon())
            active_list.append(K.cast(active, 'float32'))
            iou_list.append(iou)
    return K.sum(iou_list, axis=0) / (K.sum(active_list, axis=0) + K.epsilon())


def tpr(y_true, y_pred):
    """True positive rate。(真陽性率、再現率、Recall)

    バッチごとのtrue/falseの数が一定でない限り正しく算出されないため要注意。
    """
    mask = K.cast(K.greater_equal(y_true, 0.5), K.floatx())  # true
    pred = K.cast(K.greater_equal(y_pred, 0.5), K.floatx())  # positive
    return K.sum(pred * mask) / K.sum(mask)


def fpr(y_true, y_pred):
    """False positive rate。(偽陽性率)

    バッチごとのtrue/falseの数が一定でない限り正しく算出されないため要注意。
    """
    mask = K.cast(K.less(y_true, 0.5), K.floatx())  # false
    pred = K.cast(K.greater_equal(y_pred, 0.5), K.floatx())  # positive
    return K.sum(pred * mask) / K.sum(mask)


def fbeta_score(y_true, y_pred, beta=1):
    """Fβ-score。"""
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    axis = list(range(1, K.ndim(y_true)))
    tp = K.sum(y_true * y_pred, axis=axis)
    p = K.sum(y_pred, axis=axis)
    t = K.sum(y_true, axis=axis)
    prec = tp / (p + K.epsilon())
    rec = tp / (t + K.epsilon())
    return ((1 + beta ** 2) * prec * rec) / ((beta ** 2) * prec + rec + K.epsilon())


# 再現率(recall)
recall = tpr

# 長いので名前変えちゃう
binary_accuracy.__name__ = 'acc'
binary_iou.__name__ = 'iou'
categorical_iou.__name__ = 'iou'
