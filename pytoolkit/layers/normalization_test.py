import numpy as np
import pytest

import pytoolkit as tk

from .misc_test import _predict_layer


def test_GroupNormalization():
    _predict_layer(tk.layers.GroupNormalization(), np.zeros((1, 8, 8, 32)))
    _predict_layer(tk.layers.GroupNormalization(), np.zeros((1, 8, 8, 64)))
    _predict_layer(tk.layers.GroupNormalization(), np.zeros((1, 8, 8, 8, 32)))
    _predict_layer(tk.layers.GroupNormalization(), np.zeros((1, 8, 8, 8, 64)))


def test_InstanceNormalization():
    # [0, 1, 0, 1] => [-1, 1, -1, 1]
    # [0, 2, 0, 2] => [-1, 1, -1, 1]
    # [1, 2, 3, 4] => [-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079]
    x = np.concatenate(
        [
            np.array([[0, 1, 0, 1], [0, 2, 0, 2], [1, 2, 3, 4]]).reshape((3, 4, 1)),
            np.array([[0, 2, 0, 2], [1, 2, 3, 4], [0, 1, 0, 1]]).reshape((3, 4, 1)),
        ],
        axis=-1,
    )
    assert x.shape == (3, 4, 2)
    xm = np.mean(x, axis=1, keepdims=True)
    xs = np.std(x, axis=1, keepdims=True)
    y = (x - xm) / (xs + 1e-5)

    layer = tk.layers.InstanceNormalization(epsilon=1e-5)
    pred = _predict_layer(layer, x)
    assert pred == pytest.approx(y), (
        f"\n y[:, :, 0] = {y[:, :, 0]}"
        f"\n pred[:, :, 0] = {pred[:, :, 0]}"
        f"\n y[:, :, 1] = {y[:, :, 1]}"
        f"\n pred[:, :, 1] = {pred[:, :, 1]}"
    )
