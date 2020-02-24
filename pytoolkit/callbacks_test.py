import pathlib

import numpy as np
import pytest
import tensorflow as tf

import pytoolkit as tk


def test_ErrorOnNaN(tmpdir):
    inputs = x = tf.keras.layers.Input((32, 32, 3))
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.models.Model(inputs, x)
    model.weights[0].assign(np.array([0.5, 1.5, np.inf]))

    save_path = str(tmpdir / "___broken___.h5")
    cb = tk.callbacks.ErrorOnNaN(save_path=save_path)
    cb.model = model
    with pytest.raises(RuntimeError):
        cb.on_batch_end(3, logs={"loss": np.nan})

    assert pathlib.Path(save_path).exists()
