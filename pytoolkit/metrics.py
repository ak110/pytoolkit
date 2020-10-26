"""Kerasのmetrics関連。"""

import numpy as np
import tensorflow as tf

import pytoolkit as tk

K = tf.keras.backend


@tk.backend.name_scope
def binary_accuracy(y_true, y_pred):
    """Soft-targetとかでも一応それっぽい値を返すbinary accuracy。

    y_true = 0 or 1 で y_pred = 0.5 のとき、BCEは np.log1p(1) になる。(0.693くらい)
    それ以下なら合ってることにする。

    <https://www.wolframalpha.com/input/?dataset=&i=-(y*log(x)%2B(1-y)*log(1-x))%3Dlog(exp(0)%2B1)>

    """
    loss = tk.losses.binary_crossentropy(y_true, y_pred, reduce_mode=None)
    th = np.log1p(1)  # 0.6931471805599453
    return tk.losses.reduce(K.cast(loss < th, "float32"), reduce_mode="mean")


@tk.backend.name_scope
def binary_iou(y_true, y_pred, target_classes=None, threshold=0.5):
    """画像ごとクラスごとのIoUを算出して平均するmetric。

    Args:
        target_classes: 対象のクラスindexの配列。Noneなら全クラス。
        threshold: 予測の閾値

    """
    if target_classes is not None:
        y_true = y_true[..., target_classes]
        y_pred = y_pred[..., target_classes]
    axes = list(range(1, K.ndim(y_true)))
    t = y_true >= 0.5
    p = y_pred >= threshold
    inter = K.sum(K.cast(tf.math.logical_and(t, p), "float32"), axis=axes)
    union = K.sum(K.cast(tf.math.logical_or(t, p), "float32"), axis=axes)
    return inter / K.maximum(union, 1)


@tk.backend.name_scope
def categorical_iou(y_true, y_pred, target_classes=None, strict=True):
    """画像ごとクラスごとのIoUを算出して平均するmetric。

    Args:
        target_classes: 対象のクラスindexの配列。Noneなら全クラス。
        strict: ラベルに無いクラスを予測してしまった場合に減点されるようにするならTrue、ラベルにあるクラスのみ対象にするならFalse。

    """
    axes = list(range(1, K.ndim(y_true)))
    y_classes = K.argmax(y_true, axis=-1)
    p_classes = K.argmax(y_pred, axis=-1)
    active_list = []
    iou_list = []
    for c in target_classes or range(y_true.shape[-1]):
        with tf.name_scope(f"class_{c}"):
            y_c = K.equal(y_classes, c)
            p_c = K.equal(p_classes, c)
            inter = K.sum(K.cast(tf.math.logical_and(y_c, p_c), "float32"), axis=axes)
            union = K.sum(K.cast(tf.math.logical_or(y_c, p_c), "float32"), axis=axes)
            active = union > 0 if strict else K.any(y_c, axis=axes)
            iou = inter / (union + K.epsilon())
            active_list.append(K.cast(active, "float32"))
            iou_list.append(iou)
    return K.sum(iou_list, axis=0) / (K.sum(active_list, axis=0) + K.epsilon())


@tk.backend.name_scope
def tpr(y_true, y_pred):
    """True positive rate。(真陽性率、再現率、Recall)"""
    axes = list(range(1, K.ndim(y_true)))
    mask = K.cast(K.greater_equal(y_true, 0.5), K.floatx())  # true == 1
    pred = K.cast(K.greater_equal(y_pred, 0.5), K.floatx())  # pred == 1
    return K.sum(pred * mask, axis=axes) / K.sum(mask, axis=axes)


@tk.backend.name_scope
def tnr(y_true, y_pred):
    """True negative rate。(真陰性率)"""
    axes = list(range(1, K.ndim(y_true)))
    mask = K.cast(K.less(y_true, 0.5), K.floatx())  # true == 0
    pred = K.cast(K.less(y_pred, 0.5), K.floatx())  # pred == 0
    return K.sum(pred * mask, axis=axes) / K.sum(mask, axis=axes)


@tk.backend.name_scope
def fbeta_score(y_true, y_pred, beta=1):
    """Fβ-score。"""
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    axes = list(range(1, K.ndim(y_true)))
    tp = K.sum(y_true * y_pred, axis=axes)
    p = K.sum(y_pred, axis=axes)
    t = K.sum(y_true, axis=axes)
    prec = tp / (p + K.epsilon())
    rec = tp / (t + K.epsilon())
    return ((1 + beta ** 2) * prec * rec) / ((beta ** 2) * prec + rec + K.epsilon())


@tk.backend.name_scope
def bboxes_iou(y_true, y_pred, epsilon=1e-7):
    """bounding boxesのIoU。

    Args:
        y_true: 答えのbboxes。shape=(samples, anchors_h, anchors_w, 4)
        y_pred: 出力のbboxes。shape=(samples, anchors_h, anchors_w, 4)

    Returns:
        IoU値。shape=(samples, anchors_h, anchors_w)

    """
    inter_lt = tf.math.maximum(y_true[..., :2], y_pred[..., :2])  # 左と上
    inter_rb = tf.math.minimum(y_true[..., 2:], y_pred[..., 2:])  # 右と下
    wh_true = tf.math.maximum(y_true[..., 2:] - y_true[..., :2], 0.0)
    wh_pred = tf.math.maximum(y_pred[..., 2:] - y_pred[..., :2], 0.0)

    has_area = tf.math.reduce_all(inter_lt < inter_rb, axis=-1)
    area_inter = tf.math.reduce_prod(inter_rb - inter_lt, axis=-1) * tf.cast(
        has_area, tf.float32
    )
    area_a = tf.math.reduce_prod(wh_true, axis=-1)
    area_b = tf.math.reduce_prod(wh_pred, axis=-1)
    area_union = area_a + area_b - area_inter
    iou = area_inter / (area_union + epsilon)
    return iou


# 省略名・別名
recall = tpr

# 長いので名前変えちゃう
binary_accuracy.__name__ = "safe_acc"
binary_iou.__name__ = "iou"
categorical_iou.__name__ = "iou"


class CosineSimilarity(tf.keras.metrics.Metric):
    """コサイン類似度。"""

    def __init__(self, from_logits=False, axis=-1, name="cs", **kwargs):
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.axis = axis
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self, y_true, y_pred, sample_weight=None
    ):  # pylint: disable=arguments-differ
        """指標の算出。"""
        del sample_weight
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=self.axis)
        y_true = tf.math.l2_normalize(y_true, axis=self.axis)
        y_pred = tf.math.l2_normalize(y_pred, axis=self.axis)
        cs = tf.math.reduce_sum(y_true * y_pred, axis=self.axis)
        cs = tf.math.reduce_mean(cs)
        batch_size = tf.cast(tf.shape(y_true)[0], y_true.dtype)
        self.total.assign_add(cs * batch_size)
        self.count.assign_add(batch_size)

    def result(self):
        return self.total / tf.math.maximum(self.count, 1)

    def get_config(self):
        config = {"from_logits": self.from_logits, "axis": self.axis}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
