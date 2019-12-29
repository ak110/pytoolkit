import numpy as np
import pytest
import tensorflow as tf

import pytoolkit as tk

K = tf.keras.backend


@pytest.mark.parametrize("color", ["rgb", "lab", "hsv", "yuv", "ycbcr", "hed", "yiq"])
def test_ConvertColor(color):
    import skimage.color

    rgb = np.array(
        [
            [[0, 0, 0], [128, 128, 128], [255, 255, 255]],
            [[192, 0, 0], [0, 192, 0], [0, 0, 192]],
            [[64, 64, 0], [0, 64, 64], [64, 0, 64]],
        ],
        dtype=np.uint8,
    )

    expected = {
        "rgb": lambda rgb: rgb / 127.5 - 1,
        "lab": lambda rgb: skimage.color.rgb2lab(rgb) / 100,
        "hsv": skimage.color.rgb2hsv,
        "yuv": skimage.color.rgb2yuv,
        "ycbcr": lambda rgb: skimage.color.rgb2ycbcr(rgb) / 255,
        "hed": skimage.color.rgb2hed,
        "yiq": skimage.color.rgb2yiq,
    }[color](rgb)

    layer = tk.layers.ConvertColor(f"rgb_to_{color}")
    actual = layer(K.constant(np.expand_dims(rgb, 0))).numpy()[0]

    actual, expected = np.round(actual, 3), np.round(expected, 3)  # 丸めちゃう
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    assert actual == pytest.approx(expected, 1e-3)


def test_ChannelPair2D():
    _predict_layer(tk.layers.ChannelPair2D(), np.zeros((1, 8, 8, 3)))


def test_mixfeat():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

    x = inputs = tf.keras.layers.Input(shape=(2,))
    x = tk.layers.MixFeat()(x)
    model = tf.keras.models.Model(inputs, x)
    assert model.predict(X) == pytest.approx(X)
    model.compile("sgd", loss="mse")
    model.fit(X, X, batch_size=3, epochs=1)


def test_DropActivation():
    _predict_layer(tk.layers.DropActivation(), np.zeros((1, 8, 8, 3)))


def test_ScaleGradient():
    x = np.array([[1, 2], [3, 4]])
    assert _predict_layer(tk.layers.ScaleGradient(scale=0.1), x) == pytest.approx(x)


def _predict_layer(layer, X):
    """単一のレイヤーのModelを作って予測を行う。"""
    if isinstance(X, list):
        inputs = [tf.keras.layers.Input(shape=x.shape[1:]) for x in X]
    else:
        inputs = tf.keras.layers.Input(shape=X.shape[1:])
    model = tf.keras.models.Model(inputs=inputs, outputs=layer(inputs))
    outputs = model.predict(X)
    if isinstance(outputs, list):
        assert isinstance(model.output_shape, list)
        for o, os in zip(outputs, model.output_shape):
            assert o.shape == (len(X[0]),) + os[1:]
    else:
        assert outputs.shape == (len(X),) + model.output_shape[1:]
    return outputs
