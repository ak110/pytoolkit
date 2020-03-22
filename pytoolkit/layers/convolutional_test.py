import numpy as np
import pytest

import pytoolkit as tk

from .misc_test import _predict_layer


def test_coord_channel_2d():
    X = np.zeros((1, 4, 4, 1))
    y = np.array(
        [
            [
                [[0, 0.00, 0.00], [0, 0.25, 0.00], [0, 0.50, 0.00], [0, 0.75, 0.00]],
                [[0, 0.00, 0.25], [0, 0.25, 0.25], [0, 0.50, 0.25], [0, 0.75, 0.25]],
                [[0, 0.00, 0.50], [0, 0.25, 0.50], [0, 0.50, 0.50], [0, 0.75, 0.50]],
                [[0, 0.00, 0.75], [0, 0.25, 0.75], [0, 0.50, 0.75], [0, 0.75, 0.75]],
            ]
        ]
    )
    pred = _predict_layer(tk.layers.CoordChannel2D(), X)
    assert pred == pytest.approx(y)


def test_WSConv2D():
    _predict_layer(
        tk.layers.WSConv2D(filters=2, kernel_size=3, padding="same"),
        np.zeros((1, 8, 8, 3)),
    )


def test_RMSConv2D():
    _predict_layer(
        tk.layers.RMSConv2D(filters=2, kernel_size=3, padding="same"),
        np.zeros((1, 8, 8, 3)),
    )
