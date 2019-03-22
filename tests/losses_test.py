
import numpy as np
import pytest

import tensorflow as tf
import pytoolkit as tk

K = tk.K


def test_lovasz_hinge(dl_session):
    y_true = tf.constant([[0, 0, 0, 0], [1, 1, 1, 1]], dtype='float32')
    y_pred = tf.constant([[0, 0.25, 0.75, 1], [0, 0.25, 0.75, 1]], dtype='float32')
    loss1 = K.eval(tk.losses.lovasz_hinge(y_true, y_true))
    loss2 = K.eval(tk.losses.lovasz_hinge(y_true, y_pred))
    assert loss1 == pytest.approx([0, 0], abs=1e-5)
    assert (loss2 > np.array([0, 0])).all()
    assert loss2[0] > loss2[1]


def test_lovasz_softmax(dl_session):
    y_true = tf.constant([[[0, 1], [0, 1], [0, 1], [0, 1]]], dtype='float32')
    y_pred = tf.constant([[[0, 1], [0.25, 0.75], [0.75, 0.25], [1, 0]]], dtype='float32')
    loss1 = K.eval(tk.losses.lovasz_softmax(y_true, y_true))
    loss2 = K.eval(tk.losses.lovasz_softmax(y_true, y_pred))
    assert loss1 == pytest.approx([0], abs=1e-5)
    assert (loss2 > np.array([0])).all()
