import numpy as np
import pytest
import tensorflow as tf

import pytoolkit as tk

from .misc_test import _predict_layer


def test_ParallelGridPooling2D():
    layer = tf.keras.models.Sequential(
        [tk.layers.ParallelGridPooling2D(), tk.layers.ParallelGridGather(r=2 * 2)]
    )
    pred = _predict_layer(layer, np.zeros((1, 8, 8, 3)))
    assert pred.shape == (1, 4, 4, 3)


def test_SubpixelConv2D():
    X = np.array(
        [[[[11, 21, 12, 22], [31, 41, 32, 42]], [[13, 23, 14, 24], [33, 43, 34, 44]]]],
        dtype=np.float32,
    )
    y = np.array(
        [
            [
                [[11], [21], [31], [41]],
                [[12], [22], [32], [42]],
                [[13], [23], [33], [43]],
                [[14], [24], [34], [44]],
            ]
        ],
        dtype=np.float32,
    )
    pred = _predict_layer(tk.layers.SubpixelConv2D(scale=2), X)
    assert pred == pytest.approx(y)


def test_BlurPooling2D():
    X = np.zeros((1, 5, 5, 2))
    X[0, 2, 2, 0] = 1
    y = np.array(
        [
            [
                [[0.09765625, 0.0], [0.58593750, 0.0], [0.09765625, 0.0]],
                [[0.58593750, 0.0], [3.51562500, 0.0], [0.58593750, 0.0]],
                [[0.09765625, 0.0], [0.58593750, 0.0], [0.09765625, 0.0]],
            ]
        ]
    )
    pred = _predict_layer(tk.layers.BlurPooling2D(taps=5), X)
    assert pred == pytest.approx(y, abs=1e-5)


def test_GeMPooling2D():
    X = np.zeros((1, 8, 8, 3))
    X[0, 2, 2, 0] = 1.234 * (64 ** (1 / 3))
    y = np.array([[1.234, 0, 0]])
    pred = _predict_layer(tk.layers.GeMPooling2D(), X)
    assert pred == pytest.approx(y, abs=1e-5)
