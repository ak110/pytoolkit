import numpy as np
import pytest
import tensorflow as tf

import pytoolkit as tk


@pytest.mark.parametrize("distribute", [False, True])
@pytest.mark.parametrize("trainable", [False, True])
def test_SyncBatchNormalization(distribute, trainable):
    if distribute:
        # pylint: disable=protected-access
        old_flag, tk.hvd._initialized = tk.hvd._initialized, False
        try:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                _test_SyncBatchNormalization(trainable)
        finally:
            tk.hvd._initialized = old_flag
    else:
        tk.hvd.init()
        _test_SyncBatchNormalization(trainable)


def _test_SyncBatchNormalization(trainable):
    X = np.ones((3, 7, 7, 15))

    layer = tk.layers.SyncBatchNormalization()
    layer.trainable = trainable
    inputs = tf.keras.layers.Input((None, None, 15))
    x = layer(inputs)
    model = tf.keras.models.Model(inputs, x)
    model.compile(tf.keras.optimizers.SGD(1e-3), "mse")

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
    if trainable:
        assert gamma == pytest.approx(np.ones((15,)))
        assert beta == pytest.approx(np.ones((15,)) * 1e-4, abs=1e-4)
        assert moving_mean == pytest.approx(np.ones((15,)) * 0.01)
        assert moving_variance == pytest.approx(np.ones((15,)) * 0.99)
    else:
        assert gamma == pytest.approx(np.ones((15,)))
        assert beta == pytest.approx(np.zeros((15,)))
        assert moving_mean == pytest.approx(np.zeros((15,)))
        assert moving_variance == pytest.approx(np.ones((15,)))
