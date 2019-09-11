import pathlib

import numpy as np
import pytest

import pytoolkit as tk


@pytest.mark.usefixtures("session")
def test_xor(tmpdir):
    """XORを学習してみるコード。"""
    models_dir = pathlib.Path(str(tmpdir))
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)
    train_set = tk.data.Dataset(X.repeat(4096, axis=0), y.repeat(4096, axis=0))

    inputs = x = tk.keras.layers.Input(shape=(2,))
    x = tk.keras.layers.Dense(16, use_bias=False)(x)
    x = tk.keras.layers.BatchNormalization()(x)
    x = tk.layers.DropActivation()(x)
    x = tk.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)
    tk.models.compile(
        model, "adam", "binary_crossentropy", [tk.metrics.binary_accuracy]
    )
    tk.training.check(model)
    tk.training.train(
        model,
        train_set,
        tk.data.Preprocessor(),
        epochs=8,
        verbose=2,
        model_path=models_dir / "model.h5",
        callbacks=[
            tk.callbacks.LearningRateStepDecay(),
            tk.callbacks.CosineAnnealing(),
            tk.callbacks.TSVLogger(models_dir / "history.tsv"),
            tk.callbacks.FreezeBNCallback(1),
            tk.callbacks.UnfreezeCallback(0.0001),
            tk.callbacks.Checkpoint(models_dir / "checkpoint.h5"),
        ],
    )

    proba = tk.models.predict(model, tk.data.Dataset(X, y), tk.data.Preprocessor())
    tk.evaluations.print_classification_metrics(y, proba)

    y_pred = np.squeeze((proba > 0.5).astype(np.int32), axis=-1)
    assert (y_pred == y).all()
