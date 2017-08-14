import pathlib

import numpy as np
import pytest

import pytoolkit as tk


def test_destandarization_layer():
    import keras

    Destandarization = tk.dl.destandarization_layer_factory()
    inp = x = keras.layers.Input(shape=(1,))
    x = Destandarization(3, 5)(x)
    model = keras.models.Model(inputs=inp, outputs=x)
    pred = model.predict(np.array([[2]]))
    assert pred[0] == pytest.approx(2 * 5 + 3)


def test_weighted_mean_layer():
    import keras

    WeightedMean = tk.dl.weighted_mean_layer_factory()
    inp = x = [
        keras.layers.Input(shape=(1,)),
        keras.layers.Input(shape=(1,)),
        keras.layers.Input(shape=(1,))
    ]
    x = WeightedMean()(x)
    model = keras.models.Model(inputs=inp, outputs=x)
    pred = model.predict([np.array([1]), np.array([2]), np.array([3])])
    assert pred[0] == pytest.approx((1 + 2 + 3) / 3)


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


def test_get_custom_objects():
    custom_objects = tk.dl.get_custom_objects()
    assert str(custom_objects['Destandarization']) == str(tk.dl.destandarization_layer_factory())
    assert str(custom_objects['WeightedMean']) == str(tk.dl.weighted_mean_layer_factory())


def test_params(tmpdir):
    filepath = str(tmpdir.join('model.params.png'))

    import keras
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(32, input_shape=(16,)))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.BatchNormalization())
    tk.dl.plot_model_params(model, filepath)

    assert pathlib.Path(filepath).is_file()  # とりあえず存在チェックだけ

    assert tk.dl.count_trainable_params(model) == (16 * 32 + 32) + (32 * 32 + 32) + 32 * 2
