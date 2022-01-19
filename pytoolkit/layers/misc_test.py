import numpy as np
import pytest
import tensorflow as tf

import pytoolkit as tk


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
