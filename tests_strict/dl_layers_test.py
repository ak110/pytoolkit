
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

    import keras
    x = inputs = keras.layers.Input(shape=(2, 2, 4))
    x = tk.dl.layers.subpixel_conv2d()(scale=2)(x)
    model = keras.models.Model(inputs, x)

    assert model.predict(X) == pytest.approx(y)
