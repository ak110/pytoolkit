import numpy as np
import pytest
import tensorflow as tf

import pytoolkit as tk


def _run(func, *args, **kwargs):
    """funcをグラフモードで実行する。"""
    return tf.function(func)(*args, **kwargs)


def test_binary_crossentropy():
    # 通常の動作確認
    _binary_loss_test(tk.losses.binary_crossentropy, symmetric=True)
    # alpha
    y_true = tf.constant([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    y_pred = tf.constant([[0.0, 0.3, 0.7, 1.0], [0.0, 0.3, 0.7, 1.0]])
    loss_na = _run(tk.losses.binary_crossentropy, y_true, y_pred).numpy()
    loss_a3 = _run(tk.losses.binary_crossentropy, y_true, y_pred, alpha=0.3).numpy()
    loss_a5 = _run(tk.losses.binary_crossentropy, y_true, y_pred, alpha=0.5).numpy()
    loss_a7 = _run(tk.losses.binary_crossentropy, y_true, y_pred, alpha=0.7).numpy()
    assert loss_na == pytest.approx(loss_a5, abs=1e-6)
    assert loss_a3[0] > loss_a3[1]
    assert loss_a7[0] < loss_a7[1]


def test_binary_focal_loss():
    _binary_loss_test(tk.losses.binary_focal_loss, symmetric=True)


def test_lovasz_hinge():
    _, loss2 = _binary_loss_test(tk.losses.lovasz_hinge, symmetric=False)
    assert loss2[0] > loss2[1]


def test_lovasz_binary_crossentropy():
    y_true = tf.constant([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    y_pred = tf.constant([[0.0, 0.3, 0.7, 1.0], [0.0, 0.3, 0.7, 1.0]])
    loss1 = _run(tk.losses.lovasz_binary_crossentropy, y_true, y_true).numpy()
    loss2 = _run(tk.losses.lovasz_binary_crossentropy, y_true, y_pred).numpy()
    assert loss1 == pytest.approx(
        [0.01, 0.01], abs=1e-3
    ), "loss(y_true, y_true) == zeros"
    assert (loss2 > np.array([0.01, 0.01])).all(), "loss(y_true, y_pred) > zeros"
    # alpha
    y_true = tf.constant([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    y_pred = tf.constant([[0.0, 0.3, 0.7, 1.0], [0.0, 0.3, 0.7, 1.0]])
    loss_na = _run(tk.losses.lovasz_binary_crossentropy, y_true, y_pred).numpy()
    loss_a3 = _run(
        tk.losses.lovasz_binary_crossentropy, y_true, y_pred, alpha=0.3
    ).numpy()
    loss_a5 = _run(
        tk.losses.lovasz_binary_crossentropy, y_true, y_pred, alpha=0.5
    ).numpy()
    loss_a7 = _run(
        tk.losses.lovasz_binary_crossentropy, y_true, y_pred, alpha=0.7
    ).numpy()
    assert loss_na == pytest.approx(loss_a5, abs=1e-6)
    assert loss_a3[0] > loss_a7[0]
    assert loss_a3[1] < loss_a7[1]


def test_categorical_focal_loss():
    _categorical_loss_test(tk.losses.categorical_focal_loss, symmetric=True)


def _binary_loss_test(loss, symmetric):
    y_true = tf.constant([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    y_pred = tf.constant([[0.0, 0.3, 0.7, 1.0], [0.0, 0.3, 0.7, 1.0]])
    loss1 = _run(loss, y_true, y_true).numpy()
    loss2 = _run(loss, y_true, y_pred).numpy()
    assert loss1 == pytest.approx([0, 0], abs=1e-6), "loss(y_true, y_true) == zeros"
    assert (loss2 > np.array([0, 0])).all(), "loss(y_true, y_pred) > zeros"
    assert (
        loss2[0] == pytest.approx(loss2[1], abs=1e-5) or not symmetric
    ), "symmetricity of loss(y_true, y_pred)"
    return loss1, loss2


def _categorical_loss_test(loss, symmetric):
    y_true = tf.constant(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        ]
    )
    y_pred = tf.constant(
        [
            [[1.0, 0.0], [0.7, 0.3], [0.3, 0.7], [0.0, 1.0]],
            [[1.0, 0.0], [0.7, 0.3], [0.3, 0.7], [0.0, 1.0]],
        ]
    )
    loss1 = _run(loss, y_true, y_true).numpy()
    loss2 = _run(loss, y_true, y_pred).numpy()
    assert loss1 == pytest.approx([0, 0], abs=1e-6), "loss(y_true, y_true) == zeros"
    assert (loss2 > np.array([0, 0])).all(), "loss(y_true, y_pred) > zeros"
    assert (
        loss2[0] == pytest.approx(loss2[1], abs=1e-5) or not symmetric
    ), "symmetricity of loss(y_true, y_pred)"
    return loss1, loss2


def test_ciou():
    y_true = tf.constant(
        [
            [
                [
                    # 一致・不一致
                    [100, 100, 300, 300],
                    [100, 100, 300, 300],
                    # ゼロ
                    [100, 100, 300, 300],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [100, 100, 300, 300],
                    [100, 100, 300, 300],
                    # 負
                    [100, 100, 300, 300],
                    [100, 100, 300, 300],
                    [100, 100, 300, 300],
                ]
            ]
        ],
        dtype=tf.float32,
    )
    y_pred = tf.constant(
        [
            [
                [
                    # 一致・不一致
                    [100, 100, 300, 300],
                    [150, 150, 250, 250],
                    # ゼロ
                    [0, 0, 0, 0],
                    [100, 100, 300, 300],
                    [0, 0, 0, 0],
                    [100, 100, 100, 300],
                    [100, 100, 300, 100],
                    # 負
                    [100, 100, 99, 300],
                    [100, 100, 300, 99],
                    [100, 100, 99, 99],
                ]
            ]
        ],
        dtype=tf.float32,
    )
    assert y_true.shape.rank == 4
    assert y_pred.shape.rank == 4
    loss = _run(tk.losses.ciou, y_true, y_pred).numpy()
    assert loss.ndim == 3
    assert not np.isnan(loss).any()
    assert not np.isinf(loss).any()
    assert loss[..., 0] == 0.0
    assert (loss[..., 1:] > 0).all()

    # scale
    loss_scaled = _run(tk.losses.ciou, y_true * 123, y_pred * 123).numpy()
    assert loss == pytest.approx(loss_scaled)
