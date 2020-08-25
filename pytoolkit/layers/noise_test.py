"""テストコード。"""
import numpy as np
import tensorflow as tf

import pytoolkit as tk


def test_MixFeat():
    model = tf.keras.models.Sequential([tk.layers.MixFeat()])
    model.compile("sgd", "mse")
    model.fit(np.ones((3, 32, 32, 3)), np.ones((3, 32, 32, 3)), epochs=1, batch_size=2)
    assert model.predict(np.ones((3, 32, 32, 3))).shape == (3, 32, 32, 3)


def test_DropBlock2D():
    model = tf.keras.models.Sequential([tk.layers.DropBlock2D()])
    model.compile("sgd", "mse")
    model.fit(np.ones((3, 32, 32, 3)), np.ones((3, 32, 32, 3)), epochs=1, batch_size=2)
    assert model.predict(np.ones((3, 32, 32, 3))).shape == (3, 32, 32, 3)
