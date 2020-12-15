import numpy as np
import pytest
import tensorflow as tf

import pytoolkit as tk


@pytest.mark.parametrize("mode", ["hdf5", "saved_model", "onnx", "tflite"])
def test_save(tmpdir, mode):
    if mode == "tflite" and tf.version.VERSION.startswith("2.4."):
        pytest.xfail()
    if mode == "onnx" and tf.version.VERSION.startswith("2.4."):
        pytest.xfail()

    path = str(tmpdir / "model")

    inputs = x = tf.keras.layers.Input((32, 32, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tk.layers.Resize2D((8, 8))(x)
    x = tk.layers.GroupNormalization()(x)
    x = tk.layers.SyncBatchNormalization()(x)
    x = tk.layers.BlurPooling2D()(x)
    x = tk.layers.GeMPooling2D()(x)
    model = tf.keras.models.Model(inputs, x)
    tk.models.save(model, path, mode=mode)

    if mode in ("hdf5", "saved_model"):
        model.set_weights([-np.ones_like(w) for w in model.get_weights()])  # -1埋め
        tk.models.load_weights(model, path)  # strict=Trueでチェック
        tk.models.load(path)


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
        iterator=tk.data.DataLoader(batch_size=2).load(dataset),
        on_batch_fn=on_batch,
    )
    if output_count == 1:
        assert isinstance(result, np.ndarray)
        assert (result == dataset.data).all()
    else:
        assert isinstance(result, list)
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
        assert isinstance(result, np.ndarray)
        assert result.shape == (2 * 3 * 3, 4, 32, 32, 3)
    else:
        assert isinstance(result, list)
        assert len(result) == output_count
        assert result[0].shape == (2 * 3 * 3, 4, 32, 32, 3)
        assert result[1].shape == (2 * 3 * 3, 4, 32, 32, 3)
