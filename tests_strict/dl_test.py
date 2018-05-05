
import numpy as np

import pytoolkit as tk


def test_xor():
    import keras

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)

    builder = tk.dl.layers.Builder()
    inp = x = keras.layers.Input(shape=(2,))
    x = builder.dense(16, use_bias=False)(x)
    x = builder.bn_act()(x)
    x = builder.dense(1, activation='sigmoid')(x)
    network = keras.models.Model(inputs=inp, outputs=x)
    gen = tk.generator.Generator()
    model = tk.dl.models.Model(network, gen, batch_size=32)
    model.compile('adam', 'binary_crossentropy', ['acc'])
    model.fit(X.repeat(4096, axis=0), y.repeat(4096, axis=0), epochs=8, verbose=2)

    proba = model.predict(X)
    tk.ml.print_classification_metrics(y, proba)

    y_pred = np.squeeze((proba > 0.5).astype(np.int32), axis=-1)
    assert (y_pred == y).all()
