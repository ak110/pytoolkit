import pathlib

import numpy as np

import pytoolkit as tk


def test_xor(dl_session, tmpdir):
    models_dir = pathlib.Path(str(tmpdir))
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)

    inputs = x = tk.keras.layers.Input(shape=(2,))
    x = tk.keras.layers.Dense(16, use_bias=False)(x)
    x = tk.keras.layers.BatchNormalization()(x)
    x = tk.layers.DropActivation()(x)
    x = tk.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)
    model.compile('adam', 'binary_crossentropy', [tk.metrics.binary_accuracy])
    model.summary()
    model.fit(
        X.repeat(4096, axis=0), y.repeat(4096, axis=0), epochs=8, verbose=2,
        callbacks=[
            tk.callbacks.LearningRateStepDecay(),
            tk.callbacks.CosineAnnealing(),
            tk.callbacks.TSVLogger(models_dir / 'history.tsv'),
            tk.callbacks.EpochLogger(),
            tk.callbacks.FreezeBNCallback(1),
            tk.callbacks.UnfreezeCallback(0.0001),
            tk.callbacks.Checkpoint(models_dir / 'checkpoint.h5'),
            tk.callbacks.TerminateOnNaN(),
        ])

    proba = model.predict(X)
    tk.ml.print_classification_metrics(y, proba)

    y_pred = np.squeeze((proba > 0.5).astype(np.int32), axis=-1)
    assert (y_pred == y).all()
