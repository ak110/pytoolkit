import numpy as np
import pytest
import tensorflow as tf

import pytoolkit as tk


@pytest.mark.parametrize("mode", ["hdf5", "saved_model", "onnx", "tflite"])
def test_save(tmpdir, mode):
    if mode == "onnx":
        import keras2onnx

        if keras2onnx.__version__ < "1.6.5":
            pytest.skip()

    path = str(tmpdir / "model")
    inputs = x = tf.keras.layers.Input((32, 32, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tk.layers.Resize2D((8, 8))(x)
    if mode != "onnx":
        x = tk.layers.GroupNormalization()(x)  # keras2onnxが未対応？
        x = tk.layers.BlurPooling2D()(x)  # keras2onnxが未対応？
        x = tk.layers.GeMPooling2D()(x)  # keras2onnxが未対応？
    model = tf.keras.models.Model(inputs, x)
    tk.models.save(model, path, mode=mode)


@pytest.mark.parametrize("output_count", [1, 2])
def test_predict_flow(output_count):
    def on_batch(model, X_batch):
        assert model is None
        if output_count == 1:
            return X_batch
        else:
            return [X_batch, X_batch]

    dataset = tk.data.Dataset(data=np.arange(5))
    result = tk.models.predict(
        model=None,
        dataset=dataset,
        data_loader=tk.data.DataLoader(batch_size=2),
        on_batch_fn=on_batch,
    )
    if output_count == 1:
        assert (result == dataset.data).all()
    else:
        assert len(result) == 2
        assert (result[0] == dataset.data).all()
        assert (result[1] == dataset.data).all()


@pytest.mark.parametrize("output_count", [1, 2])
def test_predict_on_batch_augmented(output_count):
    inputs = tf.keras.layers.Input((32, 32, 3))
    outputs = [inputs * 0 + i for i in range(output_count)]
    model = tf.keras.models.Model(inputs, outputs)

    result = tk.models.predict_on_batch_augmented(
        model,
        np.zeros((4, 32, 32, 3)),
        flip=(False, True),
        crop_size=(3, 3),
        padding_size=(8, 8),
    )
    if output_count == 1:
        assert result.shape == (2 * 3 * 3, 4, 32, 32, 3)
    else:
        assert len(result) == output_count
        assert result[0].shape == (2 * 3 * 3, 4, 32, 32, 3)
        assert result[1].shape == (2 * 3 * 3, 4, 32, 32, 3)
