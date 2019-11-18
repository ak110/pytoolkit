import pathlib

import numpy as np
import tensorflow as tf

import pytoolkit as tk


def test_keras_xor(tmpdir):
    """XORを学習してみるコード。"""
    models_dir = pathlib.Path(str(tmpdir))
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)
    train_set = tk.data.Dataset(X.repeat(4096, axis=0), y.repeat(4096, axis=0))

    def create_network() -> tf.keras.models.Model:
        inputs = x = tf.keras.layers.Input(shape=(2,))
        x = tf.keras.layers.Dense(16, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        tk.models.compile(
            model, "adam", "binary_crossentropy", [tk.metrics.binary_accuracy]
        )
        return model

    model = tk.pipeline.KerasModel(
        create_network_fn=create_network,
        nfold=1,
        train_data_loader=tk.data.DataLoader(),
        val_data_loader=tk.data.DataLoader(),
        epochs=8,
        callbacks=[
            tk.callbacks.LearningRateStepDecay(),
            tk.callbacks.CosineAnnealing(),
            tk.callbacks.Checkpoint(models_dir / "checkpoint.h5"),
        ],
        fit_params={"verbose": 2},
        models_dir=models_dir,
        model_name_format="model.h5",
        use_horovod=True,
    )
    model.check()
    model.train(train_set, train_set)

    proba = model.predict(tk.data.Dataset(X, y))[0]
    tk.evaluations.print_classification_metrics(y, proba)

    y_pred = np.squeeze((proba > 0.5).astype(np.int32), axis=-1)
    assert (y_pred == y).all()
