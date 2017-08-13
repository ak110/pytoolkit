import numpy as np
import pytoolkit as tk


def test_xor(tmpdir):
    import keras

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    lr_list = [1e-3] * 3 + [1e-4] * 1

    with tk.dl.session():
        inp = x = keras.layers.Input(shape=(2,))
        x = keras.layers.Dense(16, activation='relu')(x)
        x = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.models.Model(inputs=inp, outputs=x)
        model.compile('adam', 'binary_crossentropy', ['acc'])
        model.fit(
            X.repeat(3200, axis=0),
            y.repeat(3200, axis=0),
            epochs=len(lr_list),
            verbose=2,
            callbacks=[tk.dl.my_callback_factory()(str(tmpdir), lr_list=lr_list)])
        pred = model.predict(X)
        y_pred = (pred > 0.5).astype(np.int32).reshape(y.shape)
    assert (y_pred == y).all()
