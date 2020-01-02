import numpy as np
import pytest
import tensorflow as tf

import pytoolkit as tk


def test_binary_crossentropy():
    # 通常の動作確認
    _binary_loss_test(tk.losses.binary_crossentropy, symmetric=True)
    # alpha
    y_true = tf.constant([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    y_pred = tf.constant([[0.0, 0.3, 0.7, 1.0], [0.0, 0.3, 0.7, 1.0]])
    loss_na = tk.losses.binary_crossentropy(y_true, y_pred).numpy()
    loss_a3 = tk.losses.binary_crossentropy(y_true, y_pred, alpha=0.3).numpy()
    loss_a5 = tk.losses.binary_crossentropy(y_true, y_pred, alpha=0.5).numpy()
    loss_a7 = tk.losses.binary_crossentropy(y_true, y_pred, alpha=0.7).numpy()
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
    loss1 = tk.losses.lovasz_binary_crossentropy(y_true, y_true).numpy()
    loss2 = tk.losses.lovasz_binary_crossentropy(y_true, y_pred).numpy()
    assert loss1 == pytest.approx(
        [0.0100503, 0.0100503], abs=1e-6
    ), "loss(y_true, y_true) == zeros"
    assert (
        loss2 > np.array([0.0100503, 0.0100503])
    ).all(), "loss(y_true, y_pred) > zeros"
    # alpha
    y_true = tf.constant([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    y_pred = tf.constant([[0.0, 0.3, 0.7, 1.0], [0.0, 0.3, 0.7, 1.0]])
    loss_na = tk.losses.lovasz_binary_crossentropy(y_true, y_pred).numpy()
    loss_a3 = tk.losses.lovasz_binary_crossentropy(y_true, y_pred, alpha=0.3).numpy()
    loss_a5 = tk.losses.lovasz_binary_crossentropy(y_true, y_pred, alpha=0.5).numpy()
    loss_a7 = tk.losses.lovasz_binary_crossentropy(y_true, y_pred, alpha=0.7).numpy()
    assert loss_na == pytest.approx(loss_a5, abs=1e-6)
    assert loss_a3[0] > loss_a7[0]
    assert loss_a3[1] < loss_a7[1]


def test_lovasz_softmax():
    _binary_loss_test(tk.losses.lovasz_softmax, symmetric=True)


def _binary_loss_test(loss, symmetric):
    y_true = tf.constant([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    y_pred = tf.constant([[0.0, 0.3, 0.7, 1.0], [0.0, 0.3, 0.7, 1.0]])
    loss1 = loss(y_true, y_true).numpy()
    loss2 = loss(y_true, y_pred).numpy()
    assert loss1 == pytest.approx([0, 0], abs=1e-6), "loss(y_true, y_true) == zeros"
    assert (loss2 > np.array([0, 0])).all(), "loss(y_true, y_pred) > zeros"
    assert (
        loss2[0] == pytest.approx(loss2[1], abs=1e-5) or not symmetric
    ), "symmetricity of loss(y_true, y_pred)"
    return loss1, loss2
