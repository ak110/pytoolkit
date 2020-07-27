import numpy as np
import pytest
import tensorflow as tf

import pytoolkit as tk

from .misc_test import _predict_layer


@pytest.mark.parametrize("distribute", [False, True])
def test_SyncBatchNormalization(distribute):
    tk.hvd.init()
    if distribute:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            _test_SyncBatchNormalization()
    else:
        _test_SyncBatchNormalization()


def _test_SyncBatchNormalization():
    X = np.ones((3, 7, 7, 15))

    layer = tk.layers.SyncBatchNormalization()
    inputs = tf.keras.layers.Input((None, None, 15))
    x = layer(inputs)
    model = tf.keras.models.Model(inputs, x)
    model.compile(tf.keras.optimizers.SGD(1e-7), "mse")

    # predict (moving_mean=0, moving_variance=1)
    y = model.predict(X)
    assert y == pytest.approx(X, abs=layer.epsilon)
    gamma, beta, moving_mean, moving_variance = layer.get_weights()
    assert gamma == pytest.approx(np.ones((15,)))
    assert beta == pytest.approx(np.zeros((15,)))
    assert moving_mean == pytest.approx(np.zeros((15,)))
    assert moving_variance == pytest.approx(np.ones((15,)))

    # fit
    model.fit(X, X, epochs=1, batch_size=3)
    gamma, beta, moving_mean, moving_variance = layer.get_weights()
    assert gamma == pytest.approx(np.ones((15,)), abs=1e-7)
    assert beta == pytest.approx(np.zeros((15,)), abs=1e-7)
    assert moving_mean == pytest.approx(np.ones((15,)) * 0.01)
    assert moving_variance == pytest.approx(np.ones((15,)) * 0.99)


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
