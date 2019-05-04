
import numpy as np
import pytest

import pytoolkit as tk

K = tk.K


def test_binary_crossentropy(session):
    # 通常の動作確認
    _binary_loss_test(session, tk.losses.binary_crossentropy, symmetric=True)
    # alpha = 0.5の一致確認
    y_true = K.constant([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    y_pred = K.constant([[0.0, 0.3, 0.7, 1.0], [0.0, 0.3, 0.7, 1.0]])
    loss_na = session.run(tk.losses.binary_crossentropy(y_true, y_pred))
    loss_a = session.run(tk.losses.binary_crossentropy(y_true, y_pred, alpha=0.5))
    assert loss_na == pytest.approx(loss_a, abs=1e-6)


def test_binary_focal_loss(session):
    _binary_loss_test(session, tk.losses.binary_focal_loss, symmetric=True)


def test_lovasz_hinge(session):
    _, loss2 = _binary_loss_test(session, tk.losses.lovasz_hinge, symmetric=False)
    assert loss2[0] > loss2[1]


def test_lovasz_binary_crossentropy(session):
    y_true = K.constant([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    y_pred = K.constant([[0.0, 0.3, 0.7, 1.0], [0.0, 0.3, 0.7, 1.0]])
    loss1 = session.run(tk.losses.lovasz_binary_crossentropy(y_true, y_true))
    loss2 = session.run(tk.losses.lovasz_binary_crossentropy(y_true, y_pred))
    assert loss1 == pytest.approx([0.056002, 0.056002], abs=1e-6), 'loss(y_true, y_true) == zeros'
    assert (loss2 > np.array([0.056002, 0.056002])).all(), 'loss(y_true, y_pred) > zeros'


def test_lovasz_softmax(session):
    _binary_loss_test(session, tk.losses.lovasz_softmax, symmetric=True)


def _binary_loss_test(session, loss, symmetric):
    y_true = K.constant([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    y_pred = K.constant([[0.0, 0.3, 0.7, 1.0], [0.0, 0.3, 0.7, 1.0]])
    loss1 = session.run(loss(y_true, y_true))
    loss2 = session.run(loss(y_true, y_pred))
    assert loss1 == pytest.approx([0, 0], abs=1e-6), 'loss(y_true, y_true) == zeros'
    assert (loss2 > np.array([0, 0])).all(), 'loss(y_true, y_pred) > zeros'
    assert loss2[0] == pytest.approx(loss2[1], abs=1e-5) or not symmetric, 'symmetricity of loss(y_true, y_pred)'
    return loss1, loss2
