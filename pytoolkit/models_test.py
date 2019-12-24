import pytest
import tensorflow as tf

import pytoolkit as tk


@pytest.mark.parametrize("mode", ["hdf5", "saved_model", "onnx", "tflite"])
def test_save(tmpdir, mode):
    if mode == "onnx":
        pytest.skip("keras2onnxのtf2対応待ち")

    path = str(tmpdir / "model")
    inputs = x = tf.keras.layers.Input((32, 32, 3))
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tk.layers.BlurPooling2D()(x)
    x = tk.layers.Resize2D((8, 8))(x)
    x = tk.layers.GeM2D()(x)
    model = tf.keras.models.Model(inputs, x)
    tk.models.save(model, path, mode=mode)
