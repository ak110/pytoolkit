"""Kerasのmetrics関連。"""


def tp(y_true, y_pred):
    """True positive rate。(真陽性率、再現率、Recall)

    バッチごとのtrue/falseの数が一定でない限り正しく算出されないため要注意。
    """
    import keras.backend as K
    mask = K.cast(K.greater_equal(y_true, 0.5), K.floatx())  # true
    pred = K.cast(K.greater_equal(y_pred, 0.5), K.floatx())  # positive
    return K.sum(pred * mask) / K.sum(mask)


def fp(y_true, y_pred):
    """False positive rate。(偽陽性率)

    バッチごとのtrue/falseの数が一定でない限り正しく算出されないため要注意。
    """
    import keras.backend as K
    mask = K.cast(K.less(y_true, 0.5), K.floatx())  # false
    pred = K.cast(K.greater_equal(y_pred, 0.5), K.floatx())  # positive
    return K.sum(pred * mask) / K.sum(mask)


# 再現率(recall)
recall = tp
