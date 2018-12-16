import keras
import numpy as np
import pytest

import pytoolkit as tk


def test_depth_to_space():
    X = np.array([[
        [[11, 21, 12, 22], [31, 41, 32, 42]],
        [[13, 23, 14, 24], [33, 43, 34, 44]],
    ]], dtype=np.float32)
    y = np.array([[
        [[11], [21], [31], [41]],
        [[12], [22], [32], [42]],
        [[13], [23], [33], [43]],
        [[14], [24], [34], [44]],
    ]], dtype=np.float32)

    with tk.dl.session():
        x = inputs = keras.layers.Input(shape=(2, 2, 4))
        x = tk.dl.layers.subpixel_conv2d()(scale=2)(x)
        model = keras.models.Model(inputs, x)
        assert model.predict(X) == pytest.approx(y)


def test_coord_channel_2d():
    with tk.dl.session():
        x = inputs = keras.layers.Input(shape=(4, 4, 1))
        x = tk.dl.layers.coord_channel_2d()()(x)
        model = keras.models.Model(inputs, x)
        assert model.output_shape == (None, 4, 4, 3)
        assert model.predict(np.zeros((1, 4, 4, 1))) == pytest.approx(np.array([[
            [
                [0, 0.00, 0.00],
                [0, 0.25, 0.00],
                [0, 0.50, 0.00],
                [0, 0.75, 0.00],
            ],
            [
                [0, 0.00, 0.25],
                [0, 0.25, 0.25],
                [0, 0.50, 0.25],
                [0, 0.75, 0.25],
            ],
            [
                [0, 0.00, 0.50],
                [0, 0.25, 0.50],
                [0, 0.50, 0.50],
                [0, 0.75, 0.50],
            ],
            [
                [0, 0.00, 0.75],
                [0, 0.25, 0.75],
                [0, 0.50, 0.75],
                [0, 0.75, 0.75],
            ],
        ]]))


def test_mixfeat():
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
    ], dtype=np.float32)

    with tk.dl.session():
        x = inputs = keras.layers.Input(shape=(2,))
        x = tk.dl.layers.mixfeat()()(x)
        model = keras.models.Model(inputs, x)
        assert model.predict(X) == pytest.approx(X)
        model.compile('sgd', loss='mse')
        model.fit(X, X, batch_size=3, epochs=1)
