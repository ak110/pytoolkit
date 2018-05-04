
import numpy as np

import pytoolkit as tk


def test_xor(tmpdir):
    import keras

    logger = tk.log.get('test_xor')
    logger.addHandler(tk.log.file_handler(str(tmpdir.join('test_xor.log'))))

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)

    with tk.dl.session():
        builder = tk.dl.layers.Builder()
        inp = x = keras.layers.Input(shape=(2,))
        x = builder.dense(16, use_bias=False)(x)
        x = builder.bn_act()(x)
        x = builder.dense(1, activation='sigmoid')(x)
        network = keras.models.Model(inputs=inp, outputs=x)
        gen = tk.generator.Generator()
        model = tk.dl.models.Model(network, gen, batch_size=32)
        model.compile('adam', 'binary_crossentropy', ['acc'])
        model.fit(
            X.repeat(4096, axis=0),
            y.repeat(4096, axis=0),
            epochs=8,
            verbose=2,
            callbacks=[
                tk.dl.callbacks.learning_rate(logger_name='test_xor'),
                tk.dl.callbacks.learning_curve_plot(str(tmpdir.join('history.png'))),
                tk.dl.callbacks.tsv_logger(str(tmpdir.join('history.tsv'))),
                tk.dl.callbacks.epoch_logger('test_xor'),
                tk.dl.callbacks.freeze_bn(0.5, logger_name='test_xor'),
            ])
        proba = model.predict(X)
        y_pred = np.squeeze((proba > 0.5).astype(np.int32), axis=-1)
        tk.ml.print_classification_metrics(y, proba, print_fn=logger.info)
    assert (y_pred == y).all()
