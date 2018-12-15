"""Kerasのmetrics関連。"""


def binary_accuracy(y_true, y_pred):
    """Soft-targetとかでも一応それっぽい値を返すbinary accuracy。"""
    import tensorflow as tf
    return tf.keras.backend.mean(tf.keras.backend.equal(tf.keras.backend.round(y_true), tf.keras.backend.round(y_pred)), axis=-1)


binary_accuracy.__name__ = 'acc'  # 長いので名前変えちゃう


def mean_iou(y_true, y_pred, threhsold=0.5):
    """Mean IoU。"""
    import tensorflow as tf
    y_pred_mask = tf.to_int32(y_pred >= threhsold)
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred_mask, 2)
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def tpr(y_true, y_pred):
    """True positive rate。(真陽性率、再現率、Recall)

    バッチごとのtrue/falseの数が一定でない限り正しく算出されないため要注意。
    """
    import tensorflow as tf
    mask = tf.keras.backend.cast(tf.keras.backend.greater_equal(y_true, 0.5), tf.keras.backend.floatx())  # true
    pred = tf.keras.backend.cast(tf.keras.backend.greater_equal(y_pred, 0.5), tf.keras.backend.floatx())  # positive
    return tf.keras.backend.sum(pred * mask) / tf.keras.backend.sum(mask)


def fpr(y_true, y_pred):
    """False positive rate。(偽陽性率)

    バッチごとのtrue/falseの数が一定でない限り正しく算出されないため要注意。
    """
    import tensorflow as tf
    mask = tf.keras.backend.cast(tf.keras.backend.less(y_true, 0.5), tf.keras.backend.floatx())  # false
    pred = tf.keras.backend.cast(tf.keras.backend.greater_equal(y_pred, 0.5), tf.keras.backend.floatx())  # positive
    return tf.keras.backend.sum(pred * mask) / tf.keras.backend.sum(mask)


# 再現率(recall)
recall = tpr
