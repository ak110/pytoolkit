import numpy as np
import pytest

import pytoolkit as tk


@pytest.mark.usefixtures("session")
def test_Preprocess():
    X = np.array([[[[0, 127.5, 255]]]])
    y = np.array([[[[-1, 0, +1]]]])
    pred = _predict_layer(tk.layers.Preprocess(), X)
    assert pred == pytest.approx(y)


@pytest.mark.parametrize("color", ["rgb", "lab", "hsv", "yuv", "ycbcr", "hed", "yiq"])
def test_ConvertColor(session, color):
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
    actual = session.run(layer(tk.K.constant(np.expand_dims(rgb, 0))))[0]

    actual, expected = np.round(actual, 3), np.round(expected, 3)  # 丸めちゃう
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    assert actual == pytest.approx(expected, 1e-3)


@pytest.mark.usefixtures("session")
def test_Conv2DEx():
    _predict_layer(tk.layers.Conv2DEx(4), np.zeros((1, 8, 8, 3)))


@pytest.mark.usefixtures("session")
def test_Pad2D():
    pred = _predict_layer(tk.layers.Pad2D(1), np.zeros((1, 8, 8, 3)))
    assert pred.shape == (1, 10, 10, 3)


@pytest.mark.usefixtures("session")
def test_PadChannel2D():
    pred = _predict_layer(tk.layers.PadChannel2D(filters=4), np.zeros((1, 8, 8, 3)))
    assert pred.shape == (1, 8, 8, 3 + 4)


@pytest.mark.usefixtures("session")
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


@pytest.mark.usefixtures("session")
def test_ChannelPair2D():
    _predict_layer(tk.layers.ChannelPair2D(), np.zeros((1, 8, 8, 3)))


@pytest.mark.usefixtures("session")
def test_GroupNormalization():
    _predict_layer(tk.layers.GroupNormalization(), np.zeros((1, 8, 8, 32)))
    _predict_layer(tk.layers.GroupNormalization(), np.zeros((1, 8, 8, 64)))
    _predict_layer(tk.layers.GroupNormalization(), np.zeros((1, 8, 8, 8, 32)))
    _predict_layer(tk.layers.GroupNormalization(), np.zeros((1, 8, 8, 8, 64)))


@pytest.mark.usefixtures("session")
def test_InstanceNormalization():
    # [0, 1, 0, 1] => [-1, 1, -1, 1]
    # [0, 2, 0, 2] => [-1, 1, -1, 1]
    # [1, 2, 3, 4] => [-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079]
    x = np.concatenate(
        [
            np.array([[0, 1, 0, 1], [0, 2, 0, 2], [1, 2, 3, 4]]).reshape(3, 4, 1),
            np.array([[0, 2, 0, 2], [1, 2, 3, 4], [0, 1, 0, 1]]).reshape(3, 4, 1),
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


@pytest.mark.usefixtures("session")
def test_mixfeat():
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

    x = inputs = tk.keras.layers.Input(shape=(2,))
    x = tk.layers.MixFeat()(x)
    model = tk.keras.models.Model(inputs, x)
    assert model.predict(X) == pytest.approx(X)
    model.compile("sgd", loss="mse")
    model.fit(X, X, batch_size=3, epochs=1)


@pytest.mark.usefixtures("session")
def test_DropActivation():
    _predict_layer(tk.layers.DropActivation(), np.zeros((1, 8, 8, 3)))


@pytest.mark.usefixtures("session")
def test_ParallelGridPooling2D():
    layer = tk.keras.models.Sequential(
        [tk.layers.ParallelGridPooling2D(), tk.layers.ParallelGridGather(r=2 * 2)]
    )
    pred = _predict_layer(layer, np.zeros((1, 8, 8, 3)))
    assert pred.shape == (1, 4, 4, 3)


@pytest.mark.usefixtures("session")
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


@pytest.mark.usefixtures("session")
def test_WSConv2D():
    _predict_layer(tk.layers.WSConv2D(filters=2), np.zeros((1, 8, 8, 3)))


@pytest.mark.usefixtures("session")
def test_OctaveConv2D():
    X = [np.zeros((1, 4, 4, 8)), np.zeros((1, 8, 8, 8))]
    _predict_layer(tk.layers.OctaveConv2D(filters=4), X)


@pytest.mark.usefixtures("session")
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


@pytest.mark.usefixtures("session")
def test_ScaleGradient():
    x = np.array([[1, 2], [3, 4]])
    assert _predict_layer(tk.layers.ScaleGradient(scale=0.1), x) == pytest.approx(x)


def _predict_layer(layer, X):
    """単一のレイヤーのModelを作って予測を行う。"""
    if isinstance(X, list):
        inputs = [tk.keras.layers.Input(shape=x.shape[1:]) for x in X]
    else:
        inputs = tk.keras.layers.Input(shape=X.shape[1:])
    model = tk.keras.models.Model(inputs=inputs, outputs=layer(inputs))
    outputs = model.predict(X)
    if isinstance(outputs, list):
        assert isinstance(model.output_shape, list)
        for o, os in zip(outputs, model.output_shape):
            assert o.shape == (len(X[0]),) + os[1:]
    else:
        assert outputs.shape == (len(X),) + model.output_shape[1:]
    return outputs
