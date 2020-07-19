"""Kerasの損失関数を実装するための関数など。"""

import numpy as np
import tensorflow as tf

import pytoolkit as tk


@tk.backend.name_scope
def reduce(x, reduce_mode):
    """バッチ次元だけ残して合計や平均を取る。"""
    tf.debugging.assert_rank_at_least(x, 1)
    if reduce_mode is None:
        return x
    axes = list(range(1, x.shape.rank))
    if len(axes) == 0:
        return x
    return {"sum": tf.math.reduce_sum, "mean": tf.math.reduce_mean}[reduce_mode](
        x, axis=axes
    )


@tk.backend.name_scope
def empty(y_true, y_pred):
    """ダミーのloss"""
    del y_true, y_pred
    return tf.zeros((), dtype=tf.float32)


@tk.backend.name_scope
def binary_crossentropy(
    y_true,
    y_pred,
    from_logits=False,
    alpha=None,
    label_smoothing=None,
    reduce_mode="sum",
):
    """クラス間のバランス補正ありのbinary_crossentropy。

    Args:
        alpha (float or None): class 1の重み。

    """
    assert alpha is None or 0 <= alpha <= 1

    if not from_logits:
        y_pred = tk.backend.logit(y_pred)

    if label_smoothing is not None:
        y_true = (1 - label_smoothing) * y_true + label_smoothing * 0.5

    # 前提知識:
    # -log(sigmoid(x)) = log(1 + exp(-x))
    #                  = -x + log(exp(x) + 1)
    #                  = -x + log1p(exp(x))
    # -log(1 - sigmoid(x)) = log(exp(x) + 1)
    #                      = log1p(exp(x))

    if alpha is None:
        loss = tf.math.log1p(tf.math.exp(y_pred)) - y_true * y_pred
    else:
        t = 2 * alpha * y_true - alpha - y_true + 1
        loss = 2 * (t * tf.math.log1p(tf.math.exp(y_pred)) - alpha * y_true * y_pred)

    return reduce(loss, reduce_mode)


@tk.backend.name_scope
def binary_focal_loss(
    y_true, y_pred, gamma=2.0, from_logits=False, alpha=None, reduce_mode="sum"
):
    """2クラス分類用Focal Loss <https://arxiv.org/abs/1708.02002>。

    Args:
        alpha (float or None): class 1の重み。論文では0.25。

    """
    assert alpha is None or 0 <= alpha <= 1

    if from_logits:
        y_pred, y_logit = tf.math.sigmoid(y_pred), y_pred
    else:
        y_logit = tk.backend.logit(y_pred)

    y_pred_inv = tf.cast(1 - tk.backend.clip64(y_pred), y_pred.dtype)

    # 前提知識:
    # -log(sigmoid(x)) = log(1 + exp(-x))
    #                  = -x + log(exp(x) + 1)
    #                  = -x + log1p(exp(x))
    # -log(1 - sigmoid(x)) = log(exp(x) + 1)
    #                      = log1p(exp(x))

    t = tf.math.log1p(tf.math.exp(y_logit))
    loss1 = y_true * (y_pred_inv ** gamma) * (-y_logit + t)
    loss2 = (1 - y_true) * (y_pred ** gamma) * t

    if alpha is None:
        loss = loss1 + loss2
    else:
        loss = (2 * alpha) * loss1 + (2 * (1 - alpha)) * loss2

    return reduce(loss, reduce_mode)


@tk.backend.name_scope
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
    assert tf.keras.backend.image_data_format() == "channels_last"
    assert alpha is None or class_weights is None  # 両方同時の指定はNG
    if alpha is not None:
        num_classes = y_pred.shape[-1]
        cw = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (num_classes - 1))
    elif class_weights is not None:
        assert len(class_weights) == y_pred.shape[-1]
        cw = class_weights
    else:
        cw = 1

    if from_logits:
        log_p = tk.backend.log_softmax(y_pred)
    else:
        y_pred = tf.math.maximum(y_pred, 1e-7)
        log_p = tf.math.log(y_pred)

    loss = -tf.math.reduce_sum(y_true * log_p * cw, axis=-1)

    if label_smoothing is not None:
        kl = -tf.math.reduce_mean(log_p, axis=-1)
        loss = (1 - label_smoothing) * loss + label_smoothing * kl

    return reduce(loss, reduce_mode)


@tk.backend.name_scope
def categorical_focal_loss(
    y_true,
    y_pred,
    from_logits=False,
    gamma=2.0,
    alpha=None,
    class_weights=None,
    reduce_mode="sum",
):
    """多クラス分類用Focal Loss <https://arxiv.org/abs/1708.02002>。

    Args:
        alpha (float or None): class 0以外の重み。論文では0.25。

    """
    assert tf.keras.backend.image_data_format() == "channels_last"
    assert alpha is None or class_weights is None  # 両方同時の指定はNG
    if alpha is not None:
        num_classes = y_pred.shape[-1]
        cw = np.array([(1 - alpha) * 2] * 1 + [alpha * 2] * (num_classes - 1))
    elif class_weights is not None:
        assert len(class_weights) == y_pred.shape[-1]
        cw = class_weights
    else:
        cw = 1

    if from_logits:
        log_p = tk.backend.log_softmax(y_pred)
        p = tf.nn.softmax(y_pred)
    else:
        p = y_pred
        log_p = tf.math.log(tf.math.maximum(p, 1e-7))

    w = (1 - p) ** gamma
    loss = -tf.math.reduce_sum(y_true * w * log_p * cw, axis=-1)
    return reduce(loss, reduce_mode)


@tk.backend.name_scope
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


@tk.backend.name_scope
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

        return tf.map_fn(loss_per_sample, (y_true, y_pred), dtype=y_pred.dtype)

    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))
    signs = y_true * 2.0 - 1.0  # -1 ～ +1
    errors = 1.0 - y_pred * signs
    errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0])
    weights = tk.backend.lovasz_weights(y_true, perm)
    if activation == "relu":
        errors_sorted = tf.nn.relu(errors_sorted)
    elif activation == "elu+1":
        errors_sorted = tf.nn.elu(errors_sorted) + 1
    else:
        raise ValueError(f"Invalid activation: {activation}")
    loss = tf.tensordot(errors_sorted, weights, 1)
    tf.debugging.assert_rank(loss, 0)
    return loss


@tk.backend.name_scope
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

        return tf.map_fn(loss_per_sample, (y_true, y_pred), dtype=y_pred.dtype)

    y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.reshape(y_pred, (-1,))
    if from_logits:
        lpsilon = tk.math.logit(epsilon)
        logits = tf.clip_by_value(y_pred, lpsilon, -lpsilon)
        y_pred = tf.math.sigmoid(y_pred)
    else:
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        logits = tk.backend.logit(y_pred)
    base_errors = tf.math.abs(y_true - y_pred)
    errors = tf.math.log1p(tf.math.exp(logits)) - y_true * logits  # bce
    _, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0])
    errors_sorted = tf.gather(base_errors, perm)
    weights = tk.backend.lovasz_weights(y_true, perm, alpha=alpha)
    loss = tf.tensordot(errors_sorted, weights, 1)
    tf.debugging.assert_rank(loss, 0)
    return loss


@tk.backend.name_scope
def l1_smooth_loss(y_true, y_pred):
    """L1-smooth loss。"""
    abs_loss = tf.math.abs(y_true - y_pred)
    sq_loss = 0.5 * tf.math.square(y_true - y_pred)
    l1_loss = tf.where(abs_loss < 1.0, sq_loss, abs_loss - 0.5)
    l1_loss = tf.math.reduce_sum(l1_loss, axis=-1)
    return l1_loss


@tk.backend.name_scope
def mse(y_true, y_pred, reduce_mode="mean"):
    """mean squared error。"""
    return reduce(tf.math.square(y_pred - y_true), reduce_mode)


@tk.backend.name_scope
def mae(y_true, y_pred, reduce_mode="mean"):
    """mean absolute error。"""
    return reduce(tf.math.abs(y_pred - y_true), reduce_mode)


@tk.backend.name_scope
def rmse(y_true, y_pred, reduce_mode="mean"):
    """root mean squared error。"""
    return tf.math.sqrt(reduce(tf.math.square(y_pred - y_true), reduce_mode))


@tk.backend.name_scope
def mape(y_true, y_pred, reduce_mode="mean"):
    """mean absolute percentage error。"""
    return reduce(tf.math.abs((y_true - y_pred) / y_true), reduce_mode)


@tk.backend.name_scope
def ciou(y_true, y_pred, epsilon=1e-7):
    """Complete IoU loss <https://arxiv.org/abs/1911.08287>

    Args:
        y_true: 答えのbboxes。shape=(samples, anchors_h, anchors_w, 4)
        y_pred: 出力のbboxes。shape=(samples, anchors_h, anchors_w, 4)

    Returns:
        損失の値。shape=(samples, anchors_h, anchors_w)

    """
    lt_true = tf.math.minimum(y_true[..., :2], y_true[..., 2:])
    rb_true = tf.math.maximum(y_true[..., :2], y_true[..., 2:])
    lt_pred = tf.math.minimum(y_pred[..., :2], y_pred[..., 2:])
    rb_pred = tf.math.maximum(y_pred[..., :2], y_pred[..., 2:])
    lt_inter = tf.math.maximum(lt_true, lt_pred)
    rb_inter = tf.math.minimum(rb_true, rb_pred)
    lt_union = tf.math.minimum(lt_true, lt_pred)
    rb_union = tf.math.maximum(rb_true, rb_pred)
    wh_true = tf.math.maximum(rb_true - lt_true, 0.0)
    wh_pred = tf.math.maximum(rb_pred - lt_pred, 0.0)
    c_true = (lt_true + rb_true) / 2
    c_pred = (lt_pred + rb_pred) / 2

    with tf.name_scope("iou"):
        has_area = tf.math.reduce_all(lt_inter < rb_inter, axis=-1)
        area_inter = tf.math.reduce_prod(rb_inter - lt_inter, axis=-1) * tf.cast(
            has_area, tf.float32
        )
        area_a = tf.math.reduce_prod(wh_true, axis=-1)
        area_b = tf.math.reduce_prod(wh_pred, axis=-1)
        area_union = area_a + area_b - area_inter
        iou = area_inter / (area_union + epsilon)
        tf.debugging.assert_rank(iou, y_true.shape.rank - 1)
        tf.debugging.assert_all_finite(iou, "iou")
        tf.debugging.assert_greater_equal(iou, 0.0, "iou >= 0.0")
        tf.debugging.assert_less_equal(iou, 1.0, "iou <= 1.0")

    with tf.name_scope("distance"):
        d2 = tf.math.reduce_sum((c_true - c_pred) ** 2, axis=-1)
        c2 = tf.math.reduce_sum((rb_union - lt_union) ** 2, axis=-1)
        r_diou = d2 / (c2 + epsilon)
        tf.debugging.assert_rank(r_diou, y_true.shape.rank - 1)
        tf.debugging.assert_all_finite(r_diou, "r_diou")
        tf.debugging.assert_greater_equal(r_diou, 0.0, "r_diou >= 0.0")
        tf.debugging.assert_less_equal(r_diou, 1.0, "r_diou <= 1.0")

    with tf.name_scope("aspect_ratio"):
        atan_true = tf.math.atan2(wh_true[..., 0], wh_true[..., 1])
        atan_pred = tf.math.atan2(wh_pred[..., 0], wh_pred[..., 1])
        v = (4 / np.pi ** 2) * (atan_true - atan_pred) ** 2
        v = tf.stop_gradient(v)
        tf.debugging.assert_rank(v, y_true.shape.rank - 1)
        tf.debugging.assert_all_finite(v, "v")

    with tf.name_scope("tradeoff"):
        alpha = v / ((1 - iou) + v + epsilon)
        alpha = tf.stop_gradient(alpha)
        tf.debugging.assert_all_finite(alpha, "alpha")
        tf.debugging.assert_rank(alpha, y_true.shape.rank - 1)

    with tf.name_scope("loss"):
        loss = 1 - iou + r_diou + alpha * v
        tf.debugging.assert_all_finite(loss, "loss")
        tf.debugging.assert_rank(loss, y_true.shape.rank - 1)

    return loss
