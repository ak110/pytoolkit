
import numpy as np
import pytest

import pytoolkit as tk


@pytest.mark.parametrize('color', ['rgb', 'lab', 'hsv', 'yuv', 'ycbcr', 'hed', 'yiq'])
def test_convert_color(dl_session, color):
    import skimage.color

    rgb = np.array([
        [[0, 0, 0], [128, 128, 128], [255, 255, 255]],
        [[192, 0, 0], [0, 192, 0], [0, 0, 192]],
        [[64, 64, 0], [0, 64, 64], [64, 0, 64]],
    ], dtype=np.uint8)

    expected = {
        'rgb': lambda rgb: rgb / 127.5 - 1,
        'lab': lambda rgb: skimage.color.rgb2lab(rgb) / 100,
        'hsv': skimage.color.rgb2hsv,
        'yuv': skimage.color.rgb2yuv,
        'ycbcr': lambda rgb: skimage.color.rgb2ycbcr(rgb) / 255,
        'hed': skimage.color.rgb2hed,
        'yiq': skimage.color.rgb2yiq,
    }[color](rgb)

    layer = tk.layers.ConvertColor(f'rgb_to_{color}')
    actual = dl_session.session.run(layer(tk.K.constant(np.expand_dims(rgb, 0))))[0]

    actual, expected = np.round(actual, 3), np.round(expected, 3)  # 丸めちゃう
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    assert actual == pytest.approx(expected, 1e-3)


def test_depth_to_space(dl_session):
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

    x = inputs = tk.keras.layers.Input(shape=(2, 2, 4))
    x = tk.layers.SubpixelConv2D(scale=2)(x)
    model = tk.keras.models.Model(inputs, x)
    assert model.predict(X) == pytest.approx(y)


def test_coord_channel_2d(dl_session):
    x = inputs = tk.keras.layers.Input(shape=(4, 4, 1))
    x = tk.layers.CoordChannel2D()(x)
    model = tk.keras.models.Model(inputs, x)
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


def test_mixfeat(dl_session):
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
    ], dtype=np.float32)

    x = inputs = tk.keras.layers.Input(shape=(2,))
    x = tk.layers.MixFeat()(x)
    model = tk.keras.models.Model(inputs, x)
    assert model.predict(X) == pytest.approx(X)
    model.compile('sgd', loss='mse')
    model.fit(X, X, batch_size=3, epochs=1)
