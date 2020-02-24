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

    cb = tk.callbacks.ErrorOnNaN(save_dir=str(tmpdir))
    cb.model = model
    with pytest.raises(RuntimeError):
        cb.on_batch_end(3, logs={"loss": np.nan})

    assert pathlib.Path(str(tmpdir / "broken_model.h5")).exists()
