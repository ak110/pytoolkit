"""Kerasのmetrics関連。"""


def tp(y_true, y_pred):
    """True positive rate。(真陽性率、再現率、Recall)

    バッチごとのtrue/falseの数が一定でない限り正しく算出されないため要注意。
    """
    import keras.backend as K
    return K.mean(K.greater_equal(y_true, 0.5) * K.greater_equal(y_pred, 0.5))


def fp(y_true, y_pred):
    """False positive rate。(偽陽性率)

    バッチごとのtrue/falseの数が一定でない限り正しく算出されないため要注意。
    """
    import keras.backend as K
    return K.mean(K.less(y_true, 0.5) * K.greater_equal(y_pred, 0.5))


# 再現率(recall)
recall = tp
