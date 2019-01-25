"""Kerasのmetrics関連。"""


def binary_accuracy(y_true, y_pred):
    """Soft-targetとかでも一応それっぽい値を返すbinary accuracy。"""
    import keras.backend as K
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)), axis=-1)


binary_accuracy.__name__ = 'acc'  # 長いので名前変えちゃう


def categorical_iou(target_classes=None, strict=True):
    """画像ごとクラスごとのIoUを算出して平均するmetric。

    Args:
        target_classes: 対象のクラスindexの配列。Noneなら全クラス。
        strict: ラベルに無いクラスを予測してしまった場合に減点されるようにするならTrue、ラベルにあるクラスのみ対象にするならFalse。

    """
    def iou(y_true, y_pred):
        import keras.backend as K
        import tensorflow as tf
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
    return iou


def mean_iou(y_true, y_pred, threhsold=0.5):
    """Mean IoU。"""
    import keras.backend as K
    import tensorflow as tf
    y_pred_mask = tf.to_int32(y_pred >= threhsold)
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred_mask, 2)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def tpr(y_true, y_pred):
    """True positive rate。(真陽性率、再現率、Recall)

    バッチごとのtrue/falseの数が一定でない限り正しく算出されないため要注意。
    """
    import keras.backend as K
    mask = K.cast(K.greater_equal(y_true, 0.5), K.floatx())  # true
    pred = K.cast(K.greater_equal(y_pred, 0.5), K.floatx())  # positive
    return K.sum(pred * mask) / K.sum(mask)


def fpr(y_true, y_pred):
    """False positive rate。(偽陽性率)

    バッチごとのtrue/falseの数が一定でない限り正しく算出されないため要注意。
    """
    import keras.backend as K
    mask = K.cast(K.less(y_true, 0.5), K.floatx())  # false
    pred = K.cast(K.greater_equal(y_pred, 0.5), K.floatx())  # positive
    return K.sum(pred * mask) / K.sum(mask)


# 再現率(recall)
recall = tpr
